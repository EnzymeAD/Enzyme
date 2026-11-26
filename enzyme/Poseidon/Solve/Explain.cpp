//===- Explain.cpp - what a solve changed, considered and rejected --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Explain.h"
#include "Core/FPCore.h"
#include "Core/Session.h"
#include "Core/Technique.h"
#include "Model/CostModel.h"
#include "Pins.h"
#include "Sites/SiteIdentity.h"
#include "Solve.h"
#include "Support/Diagnostics.h"
#include "Support/Utils.h"
#include "Targets/MathLibrary.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

static constexpr size_t kMaxRejectedRemarks = 3;

static void putNum(json::Object &o, StringRef key, double v) {
  if (std::isfinite(v))
    o[key] = v;
}

std::string remarkNumber(double v, int digits) {
  if (!std::isfinite(v))
    return "n/a";
  return formatv("{0}", format("%.*g", digits, v)).str();
}

static std::string pctStr(int64_t before, int64_t after) {
  if (before == 0)
    return "";
  double pct = 100.0 * (double)(after - before) / (double)before;
  return formatv(" ({0}{1}%)", pct >= 0 ? "+" : "", format("%.0f", pct)).str();
}

std::string clip(const std::string &s, size_t n) {
  if (s.size() <= n)
    return s;
  return s.substr(0, n) + "...";
}

// The C type of T, or T as the IR prints it when it is not a float type.
static std::string typeName(Type *T) {
  std::string dtype = dtypeOfType(T);
  if (!dtype.empty())
    return cTypeName(dtype).str();
  std::string s;
  raw_string_ostream os(s);
  T->print(os);
  return s;
}

Expected<Alternative> pricedAlternative(const Explanation &E,
                                        const SiteUnit &unit, size_t i,
                                        std::optional<double> costDelta,
                                        double accDelta) {
  Alternative A;
  A.id = unit.candidateId(i);
  A.unit = &unit;
  A.index = i;
  if (costDelta) {
    Expected<int64_t> c = E.onAxis(*costDelta);
    if (!c)
      return c.takeError();
    A.costDelta = *c;
  }
  E.setAccuracy(A, accDelta);
  return A;
}

Alternative droppedAlternative(StringRef prefix, StringRef why) {
  Alternative A;
  A.standing = Standing::Dropped;
  A.reason = (prefix + why).str();
  return A;
}

std::string joinWords(ArrayRef<std::string> items) {
  std::string out;
  for (size_t k = 0; k < items.size(); ++k) {
    if (k)
      out += k + 1 == items.size() ? " and " : ", ";
    out += items[k];
  }
  return out;
}

namespace {

struct Linear {
  SmallVector<std::pair<std::string, int64_t>, 2> terms;
  int64_t c = 0;
  bool ok = true;

  Linear &add(const Linear &o, int64_t scale) {
    ok = ok && o.ok;
    c += scale * o.c;
    for (const auto &term : o.terms) {
      auto it = llvm::find_if(
          terms, [&](const auto &t) { return t.first == term.first; });
      if (it == terms.end())
        terms.push_back({term.first, scale * term.second});
      else
        it->second += scale * term.second;
    }
    llvm::erase_if(terms, [](auto &t) { return t.second == 0; });
    return *this;
  }

  bool divide(int64_t d) {
    if (d <= 0 || c % d)
      return false;
    for (auto &t : terms)
      if (t.second % d)
        return false;
    c /= d;
    for (auto &t : terms)
      t.second /= d;
    return true;
  }

  std::string str() const {
    std::string s;
    for (const auto &[name, coeff] : terms) {
      int64_t a = coeff < 0 ? -coeff : coeff;
      std::string term = a == 1 ? name : std::to_string(a) + " * " + name;
      if (s.empty())
        s = coeff < 0 ? "-" + term : term;
      else
        s += (coeff < 0 ? " - " : " + ") + term;
    }
    if (s.empty())
      return std::to_string(c);
    if (c > 0)
      s += " + " + std::to_string(c);
    else if (c < 0)
      s += " - " + std::to_string(-c);
    return s;
  }
};

class ValueNamer {
  const DataLayout &DL;
  DenseMap<const Value *, const DILocalVariable *> exact, any;
  DenseMap<const DILocalVariable *, SmallPtrSet<const Value *, 2>> bound;

  std::string anyName(const Value *V) const {
    auto it = any.find(V);
    return it == any.end() ? "" : it->second->getName().str();
  }

  Linear linear(Value *V, unsigned depth = 0) const {
    Linear L;
    if (depth > 8) {
      L.ok = false;
      return L;
    }
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      L.c = CI->getSExtValue();
      return L;
    }
    if (isa<SExtInst>(V) || isa<ZExtInst>(V) || isa<TruncInst>(V))
      return linear(cast<Instruction>(V)->getOperand(0), depth + 1);
    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      Value *A = BO->getOperand(0), *B = BO->getOperand(1);
      auto *CA = dyn_cast<ConstantInt>(A), *CB = dyn_cast<ConstantInt>(B);
      switch (BO->getOpcode()) {
      case Instruction::Add:
        return linear(A, depth + 1).add(linear(B, depth + 1), 1);
      case Instruction::Sub:
        return linear(A, depth + 1).add(linear(B, depth + 1), -1);
      case Instruction::Or:
        if (cast<PossiblyDisjointInst>(BO)->isDisjoint())
          return linear(A, depth + 1).add(linear(B, depth + 1), 1);
        break;
      case Instruction::Mul:
        if (CB)
          return Linear().add(linear(A, depth + 1), CB->getSExtValue());
        if (CA)
          return Linear().add(linear(B, depth + 1), CA->getSExtValue());
        break;
      case Instruction::Shl:
        if (CB && CB->getZExtValue() < 62)
          return Linear().add(linear(A, depth + 1),
                              (int64_t)1 << CB->getZExtValue());
        break;
      default:
        break;
      }
    }
    std::string name = anyName(V);
    if (name.empty())
      L.ok = false;
    else
      L.terms.push_back({name, 1});
    return L;
  }

  std::string baseName(const Value *V) const {
    if (auto *GV = dyn_cast<GlobalVariable>(V)) {
      SmallVector<DIGlobalVariableExpression *, 1> gves;
      GV->getDebugInfo(gves);
      for (auto *gve : gves)
        if (auto *var = gve->getVariable())
          return var->getName().str();
      return GV->getName().str();
    }
    return anyName(V);
  }

public:
  explicit ValueNamer(Function &F) : DL(F.getParent()->getDataLayout()) {
    for (Instruction &I : instructions(F))
      for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
        if (!DVR.isDbgValue() || DVR.hasArgList() || DVR.isKillLocation())
          continue;
        Value *V = DVR.getVariableLocationOp(0);
        const DILocalVariable *var = DVR.getVariable();
        if (!V || !var || var->getName().empty())
          continue;
        bound[var].insert(V);
        any.try_emplace(V, var);
        if (DVR.getExpression()->getNumElements() == 0)
          exact.try_emplace(V, var);
      }
  }

  std::string element(Value *ptr, uint64_t elemSize) const {
    Linear bytes;
    Value *base = ptr;
    while (auto *GEP = dyn_cast<GEPOperator>(base)) {
      for (gep_type_iterator GTI = gep_type_begin(GEP), E = gep_type_end(GEP);
           GTI != E; ++GTI) {
        if (GTI.getStructTypeOrNull())
          return "";
        int64_t stride =
            (int64_t)GTI.getSequentialElementStride(DL).getFixedValue();
        bytes.add(linear(GTI.getOperand()), stride);
        if (!bytes.ok)
          return "";
      }
      base = GEP->getPointerOperand();
    }
    if (base == ptr || !bytes.divide((int64_t)elemSize))
      return "";
    std::string b = baseName(base);
    if (b.empty())
      return "";
    return b + "[" + bytes.str() + "]";
  }

  std::string name(Value *V) const {
    if (auto it = exact.find(V); it != exact.end()) {
      auto b = bound.find(it->second);
      if (b != bound.end() && b->second.size() == 1)
        return it->second->getName().str();
    }
    if (auto *LI = dyn_cast<LoadInst>(V)) {
      std::string s =
          element(LI->getPointerOperand(), DL.getTypeStoreSize(LI->getType()));
      if (!s.empty())
        return s;
    }
    if (isa<Argument>(V) || isa<GlobalVariable>(V))
      return baseName(V);
    return "";
  }

  std::string outputName(Instruction *O) const {
    for (User *U : O->users())
      if (auto *SI = dyn_cast<StoreInst>(U))
        if (SI->getValueOperand() == O) {
          std::string s = element(SI->getPointerOperand(),
                                  DL.getTypeStoreSize(O->getType()));
          if (!s.empty())
            return s;
        }
    return name(O);
  }
};

} // namespace

// Adds `keys`, the keys of a kind, to `o`; json::Object prints its keys
// sorted, so the order they are added in is not written.
static void mergeKeys(json::Object &o, json::Object keys) {
  for (auto &KV : keys) {
    [[maybe_unused]] bool fresh =
        o.try_emplace(KV.first, std::move(KV.second)).second;
    assert(fresh && "a key of a kind is one the entry does not write itself");
  }
}

// The `status` of a candidate in the explanation.
static StringRef statusName(Standing standing) {
  switch (standing) {
  case Standing::Applied:
    return "applied";
  case Standing::Pareto:
    return "pareto";
  default:
    return "rejected";
  }
}

static std::string exprWords(const UnitExplanation &U) {
  if (!U.outC.empty() && !U.outC[0].empty())
    return clip(U.outC[0]);
  return formatv("the subgraph of {0} operations at {1}:{2}", U.ops.size(),
                 U.loc.line, U.loc.col)
      .str();
}

static std::string siteSymbolOf(const Session &S, const Function &F) {
  return siteSymbol(siteKey(S.get<SiteRecords>(), F)).str();
}

// The schema identifier of every explanation; see Explainer::toJSON.
static constexpr StringLiteral kSchema = "poseidon-explain/1";

// Why a site, and each unit of the point it selected, stays as written when the
// memory traffic leaves that point no faster.
static constexpr StringLiteral kNotFaster =
    "the selected frontier point is not faster than the code as written once "
    "the memory traffic is priced";

// The names of the site whose clone is F, or whose body is F when no clone was
// made, where it is called and where it is written.
static json::Object functionJSON(const Session &S, const Function &F) {
  json::Object fn;
  const Function *origin = siteOriginOf(S.get<SiteRecords>(), F);
  std::string symbol = siteSymbolOf(S, F);
  fn["name"] = llvm::demangle(symbol);
  fn["symbol"] = symbol;
  fn["ir"] = F.getName().str();
  auto site = S.SiteCalls.find(&F);
  if (site != S.SiteCalls.end())
    putLoc(fn, "site", locOf(site->second));
  const DISubprogram *SP = origin ? origin->getSubprogram() : F.getSubprogram();
  if (SP)
    putLoc(fn, "loc", SourceLoc{SP->getFilename().str(), SP->getLine(), 0});
  return fn;
}

// The target a site is solved for: `tau` > 0, from the site or from
// -poseidon-tau, or else the cost budget.
static json::Object targetJSON(const Options &O, double tau, bool siteTarget) {
  json::Object target;
  if (tau > 0.0) {
    target["tau"] = tau;
    target["tau_from"] = siteTarget ? "site" : "flag";
  } else {
    target["budget"] = O.Solver.CostBudget;
  }
  return target;
}

// Writes `doc`, the explanation of a site, as <stem>.explain.json into the
// explanation directory.
static void writeExplanation(const Session &S, const Function &F,
                             StringRef stem, json::Object doc) {
  const std::string &dir = S.Opts.Diagnostics.Explain;
  if (auto EC = sys::fs::create_directories(dir)) {
    diagnose(F,
             "cannot create the explanation directory " + dir + ": " +
                 EC.message(),
             DS_Warning);
    return;
  }
  SmallString<128> path(dir);
  sys::path::append(path, stem + ".explain.json");
  if (Error Err = writeToOutput(path, [&](raw_ostream &os) {
        os << formatv("{0:2}", json::Value(std::move(doc))) << "\n";
        return Error::success();
      })) {
    diagnose(F,
             "cannot write the explanation " + path + ": " +
                 toString(std::move(Err)),
             DS_Warning);
    return;
  }
  if (S.Opts.Diagnostics.Print)
    errs() << "[poseidon] explanation written to " << path << '\n';
}

DiagnosticLocation siteLocation(const Session &S, const Function &F) {
  auto site = S.SiteCalls.find(&F);
  return site != S.SiteCalls.end() && site->second
             ? DiagnosticLocation(site->second)
             : DiagnosticLocation(F.getSubprogram());
}

namespace {

class Explainer final : public Explanation {
  Session &Sess;
  Function &F;
  SiteState &st;
  ArrayRef<SolutionStep> steps;
  const SolveRecord &rec;
  ValueNamer namer;
  // An overlay: fpcoreToC may add to the graph it is given, and the
  // explanation must leave the solve's own as it is.
  FPGraph scratchGraph;
  SmallVector<SolutionStep> stepsApplied;
  std::set<std::pair<const SiteUnit *, size_t>> applied, skipped;
  // Per candidate on the frontier, the best verdict of the points that hold
  // it, and the cheapest of those points.
  std::map<std::pair<const SiteUnit *, size_t>, std::pair<PointVerdict, DPCost>>
      onFrontier;
  std::vector<UnitExplanation> units;
  double A0, S, tau;
  // Puts the units' cost changes on the axis the solve compared them on.
  CostScale scale;

  // A cost on the DP's axis, as the share of the profiled cost it stands for.
  std::string costStr(int64_t c) const {
    return formatv("{0}", format("%.6g", scale.share(c))).str();
  }

  static std::pair<const SiteUnit *, size_t> keyOf(const SolutionStep &step) {
    return {&unitOf(step), step.candidateIndex};
  }

  // Empty for a step of another site of a joint solve.
  std::string stepId(const SolutionStep &step) const {
    const SiteUnit &unit = unitOf(step);
    return &unit.site() == &st ? unit.candidateId(step.candidateIndex) : "";
  }

  double relErr(double acc) const {
    if (!(S > 0.0) || !std::isfinite(A0) || !std::isfinite(acc))
      return std::numeric_limits<double>::quiet_NaN();
    return relativeError(A0 + acc, S, rec.moment);
  }

  double share(double acc) const {
    if (!(S > 0.0) || !std::isfinite(acc))
      return std::numeric_limits<double>::quiet_NaN();
    return relativeError(acc, S, rec.moment);
  }

  FPNode *nodeOf(Value *V) const {
    auto it = st.valueToNodeMap.find(V);
    return it == st.valueToNodeMap.end() ? nullptr : it->second;
  }

  std::string rangeOf(Value *V, const std::string &label) {
    auto it = st.valueToNodeMap.find(V);
    if (it == st.valueToNodeMap.end() || !it->second)
      return "";
    std::optional<std::pair<double, double>> range =
        it->second->profiledRange();
    if (!range)
      return "";
    auto [lo, hi] = *range;
    int digits = 3;
    while (lo != hi && digits < 17 &&
           remarkNumber(lo, digits) == remarkNumber(hi, digits))
      ++digits;
    return label + " in [" + remarkNumber(lo, digits) + ", " +
           remarkNumber(hi, digits) + "]";
  }

  std::string leafLabel(Value *V) {
    std::string n = namer.name(V);
    if (!n.empty())
      return n;
    auto it = st.valueToNodeMap.find(V);
    if (it != st.valueToNodeMap.end() && it->second &&
        !it->second->symbol.empty())
      return it->second->symbol;
    return "";
  }

  void markSteps();
  Error buildSubgraph(size_t s);
  void judge(UnitExplanation &U);
  // A candidate on the frontier point `point` the selection passed over.
  void markPareto(Alternative &A, DPCost point) const;
  std::string leftReason(const UnitExplanation &U);
  std::optional<int64_t> appliedDelta() const;
  std::optional<int64_t> costBefore() const;
  std::optional<double> selectedAcc() const;
  // Why the solve left the site as written, when it applied no step.
  std::string siteLeftReason() const;
  // Whether the selected point was left for the code as written, which its
  // memory traffic makes no slower.
  bool selectedNotTaken() const { return rec.time && !rec.time->taken; }

public:
  Explainer(Session &Sess, Function &F, SiteState &st,
            ArrayRef<SolutionStep> steps, const SolveRecord &rec)
      : Sess(Sess), F(F), st(st), steps(steps), rec(rec), namer(F),
        scratchGraph(st.overlay()), A0(rec.baseline), S(rec.accScale),
        tau(rec.tau), scale(rec.profiledCost) {}

  Expected<int64_t> onAxis(double cost) const override;
  void setAccuracy(Alternative &A, double acc) const override {
    if (!std::isfinite(acc))
      return;
    A.accDelta = acc;
    if (double e = relErr(acc); std::isfinite(e))
      A.err = e;
  }
  CRendering rendering(const Subgraph &sg) override {
    CRendering how;
    how.inputs = &sg.inputs;
    how.leafName = [this](Value *V) { return namer.name(V); };
    return how;
  }
  FPGraph &scratch() override { return scratchGraph; }
  ArrayRef<SolutionStep> appliedSteps() const override { return stepsApplied; }

  Error build();
  void emitRemarks();
  json::Object toJSON();
};

} // namespace

void Explainer::markSteps() {
  stepsApplied = stepsToApply(steps);
  for (const SolutionStep &step : stepsApplied)
    applied.insert(keyOf(step));
  for (const SolutionStep &step : steps)
    if (!applied.count(keyOf(step)))
      skipped.insert(keyOf(step));
  for (const auto &[cost, sol] : rec.solutions) {
    const PointVerdict verdict = rec.verdicts.at(cost);
    for (const SolutionStep &step : sol) {
      auto [it, fresh] = onFrontier.try_emplace(keyOf(step), verdict, cost);
      if (!fresh && verdict < it->second.first)
        it->second = {verdict, cost};
    }
  }
}

Expected<int64_t> Explainer::onAxis(double cost) const {
  if (std::optional<DPCost> c = scale.toDPCost(cost))
    return *c;
  return userError(F, offAxisMessage("a cost of", cost));
}

void Explainer::judge(UnitExplanation &U) {
  for (Alternative &A : U.alts) {
    if (!A.unit)
      continue;
    auto key = std::make_pair(A.unit, A.index);
    if (applied.count(key)) {
      A.standing = Standing::Applied;
      U.applied = true;
      if (rec.selector == SolveSelector::Pinned)
        A.reason = "selected by -poseidon-pin";
      else if (tau > 0.0)
        A.reason = "part of the cheapest solution that meets the target";
      else
        A.reason = "part of the most accurate solution within the cost budget";
      continue;
    }
    if (skipped.count(key)) {
      A.reason = "another selected rewrite of the same output was applied "
                 "first";
      A.standing = Standing::Far;
      continue;
    }
    auto frontier = onFrontier.find(key);
    if (frontier != onFrontier.end() &&
        frontier->second.first == PointVerdict::Outranked) {
      markPareto(A, frontier->second.second);
      continue;
    }
    if (tau > 0.0 && A.err && *A.err > tau) {
      A.reason = "modelled error " + remarkNumber(*A.err) +
                 " is above the target " + remarkNumber(tau);
      A.standing = A.costDelta && *A.costDelta < 0
                       ? Standing::CheaperAboveTarget
                       : Standing::Near;
      continue;
    }
    if (A.costDelta && A.accDelta && *A.costDelta >= 0 && *A.accDelta >= 0) {
      A.reason = "neither cheaper nor more accurate than as written";
      A.standing = Standing::Far;
      continue;
    }
    if (frontier != onFrontier.end()) {
      markPareto(A, frontier->second.second);
      continue;
    }
    const Alternative *dom = nullptr;
    auto better = [&](const Alternative &B, const Alternative &D) {
      bool b = applied.count({B.unit, B.index}),
           d = applied.count({D.unit, D.index});
      return b != d ? b : *B.costDelta < *D.costDelta;
    };
    if (A.costDelta && A.accDelta)
      for (const Alternative &B : U.alts)
        if (&B != &A && B.unit && B.costDelta && B.accDelta &&
            *B.costDelta <= *A.costDelta && *B.accDelta <= *A.accDelta &&
            (*B.costDelta < *A.costDelta || *B.accDelta < *A.accDelta) &&
            (!dom || better(B, *dom)))
          dom = &B;
    if (dom) {
      A.reason = "dominated by " + dom->id +
                 ", which costs no more and is at least as accurate";
      A.standing = Standing::Far;
      continue;
    }
    std::string rival;
    for (const Alternative &B : U.alts)
      if (&B != &A && B.unit && applied.count({B.unit, B.index}) &&
          B.outputs == A.outputs)
        if (std::string why = A.unit->rivalReason(A, B); !why.empty())
          rival = std::move(why);
    if (!rival.empty()) {
      A.reason = std::move(rival);
      A.standing = Standing::Far;
      continue;
    }
    if (tau == 0.0 && A.costDelta &&
        *A.costDelta > scale.fromShare(Sess.Opts.Solver.CostBudget)) {
      A.reason = "its cost change " + costStr(*A.costDelta) +
                 " is above the budget " +
                 costStr(scale.fromShare(Sess.Opts.Solver.CostBudget));
      A.standing = Standing::Near;
      continue;
    }
    A.reason = "not on the Pareto frontier of the site";
    A.standing = A.accDelta && *A.accDelta < 0 ? Standing::Near : Standing::Far;
  }
}

void Explainer::markPareto(Alternative &A, DPCost point) const {
  std::string ids;
  for (const SolutionStep &step : rec.solutions.at(point))
    ids += (ids.empty() ? "" : ",") + stepId(step);
  A.standing = Standing::Pareto;
  A.reason = "on the Pareto frontier; -poseidon-pin=" + ids + " applies it";
}

std::string Explainer::leftReason(const UnitExplanation &U) {
  if (U.applied)
    return "";
  bool any = llvm::any_of(U.alts, [](const Alternative &A) { return A.unit; });
  if (!any) {
    SmallVector<std::string, 2> why(U.whyNone.begin(), U.whyNone.end());
    if (U.sg)
      for (const Technique *T : techniques())
        if (llvm::none_of(st.units, [&](SiteUnit *unit) {
              return &unit->info() == &T->unitKind() &&
                     unit->subgraphOf() == U.sg;
            }))
          T->explainAbsent(Sess.Opts, st, *U.sg, why);
    if (why.empty())
      why.push_back("no candidate was generated");
    llvm::sort(why);
    why.erase(std::unique(why.begin(), why.end()), why.end());
    return joinWords(why);
  }
  if (tau > 0.0) {
    const Alternative *best = nullptr;
    for (const Alternative &A : U.alts)
      if (A.unit && A.err && (!best || *A.err < *best->err))
        best = &A;
    bool anyMeets = llvm::any_of(U.alts, [&](const Alternative &A) {
      return A.unit && A.err && *A.err <= tau;
    });
    if (!anyMeets && best)
      return "every alternative's modelled error is above the target " +
             remarkNumber(tau) + " (best: " + best->id + ", " +
             remarkNumber(*best->err) + ")";
    bool cheaperMeets = llvm::any_of(U.alts, [&](const Alternative &A) {
      return A.unit && A.err && *A.err <= tau && A.costDelta &&
             *A.costDelta < 0;
    });
    if (!cheaperMeets)
      return "no alternative that meets the target " + remarkNumber(tau) +
             " is cheaper than as written";
    if (!rec.selected && steps.empty())
      return "no solution on the site's frontier meets the target " +
             remarkNumber(tau);
    if (selectedNotTaken())
      return kNotFaster.str();
    return "its alternatives do not fit the target together with the "
           "selected rewrites";
  }
  if (selectedNotTaken())
    return kNotFaster.str();
  return "no alternative lowers the modelled error within the cost budget " +
         costStr(scale.fromShare(Sess.Opts.Solver.CostBudget));
}

Error Explainer::buildSubgraph(size_t s) {
  Subgraph &sg = st.subgraphs[s];
  UnitExplanation U;
  U.id = formatv("S{0}", s).str();
  U.kind = "subgraph";
  U.sg = &sg;
  U.root = sg.outputs.empty() ? nullptr : sg.outputs[0];
  U.loc = locOf(U.root);
  CRendering how = rendering(sg);
  DenseMap<const Value *, unsigned> opIdx, inIdx;

  for (Instruction &I : instructions(F)) {
    if (!sg.operations.contains(&I))
      continue;
    opIdx[&I] = U.ops.size();
    json::Object op;
    FPNode *N = nodeOf(&I);
    const std::string opcode = N ? operationName(*N) : "";
    if (N)
      op["opcode"] = opcode;
    if (N && hasFloatSuffixInC(opcode))
      op["float_call"] = opcode + "f";
    op["type"] = typeName(I.getType());
    putLoc(op, "loc", locOf(&I));
    if (N) {
      op["executions"] = (int64_t)N->executions;
      putNum(op, "sensitivity", N->sens);
      putNum(op, "gradient", N->grad);
      if (std::optional<std::pair<double, double>> range = N->profiledRange()) {
        op["min"] = range->first;
        op["max"] = range->second;
      }
    }
    U.ops.push_back(std::move(op));
  }

  for (Value *V : sg.inputs) {
    inIdx[V] = U.inputs.size();
    json::Object in;
    FPNode *N = nodeOf(V);
    if (N && !N->symbol.empty())
      in["var"] = N->symbol;
    std::string name = namer.name(V);
    if (!name.empty())
      in["name"] = name;
    putLoc(in, "loc", locOf(V));
    in["type"] = typeName(V->getType());
    if (N) {
      if (std::optional<std::pair<double, double>> range = N->profiledRange()) {
        in["min"] = range->first;
        in["max"] = range->second;
        std::string label = leafLabel(V);
        if (!label.empty())
          U.inputRanges.push_back(rangeOf(V, label));
      }
    }
    U.inputs.push_back(std::move(in));
  }

  for (Instruction *O : sg.outputs) {
    json::Object out;
    std::string name = namer.outputName(O);
    if (!name.empty())
      out["name"] = name;
    putLoc(out, "loc", locOf(O));
    out["type"] = typeName(O->getType());
    FPNode *ON = nodeOf(O);
    SmallVector<FPNode *, 16> todo;
    SmallPtrSet<FPNode *, 16> seen;
    std::set<unsigned> coneOps, coneIns;
    if (ON)
      todo.push_back(ON);
    while (!todo.empty()) {
      FPNode *N = todo.pop_back_val();
      if (!seen.insert(N).second)
        continue;
      auto *LL = dyn_cast<FPLLValue>(N);
      if (!LL)
        continue;
      if (auto in = inIdx.find(LL->value); in != inIdx.end()) {
        coneIns.insert(in->second);
        continue;
      }
      auto op = opIdx.find(LL->value);
      if (op == opIdx.end())
        continue;
      coneOps.insert(op->second);
      for (const auto &operand : N->operands)
        todo.push_back(operand);
    }
    json::Array cops, cins;
    for (unsigned k : coneOps)
      cops.push_back((int64_t)k);
    for (unsigned k : coneIns)
      cins.push_back((int64_t)k);
    out["ops"] = std::move(cops);
    out["inputs"] = std::move(cins);
    std::string c = ON ? renderC(*ON, how, dtypeOfType(O->getType())) : "";
    if (!c.empty())
      out["c"] = c;
    U.outC.push_back(c);
    if (ON) {
      U.executions = std::max(U.executions, ON->executions);
      if (std::isfinite(ON->sens))
        U.sensitivity =
            std::isfinite(U.sensitivity) ? U.sensitivity + ON->sens : ON->sens;
    }
    U.outputs.push_back(std::move(out));
  }

  if (!sg.outputs.empty()) {
    SmallVector<Value *> outs(sg.outputs.begin(), sg.outputs.end());
    Expected<double> base =
        compCost(Sess.get<CostModelState>(), outs, sg.inputs);
    if (!base)
      return base.takeError();
    if (FPNode *N = nodeOf(sg.outputs[0])) {
      Expected<int64_t> c = onAxis(*base * (double)N->executions);
      if (!c)
        return c.takeError();
      U.baseCost = *c;
    }
  }

  for (SiteUnit *unit : st.units)
    if (unit->subgraphOf() == &sg)
      if (Error E = unit->explain(*this, U))
        return E;
  judge(U);
  U.left = leftReason(U);
  units.push_back(std::move(U));
  return Error::success();
}

std::optional<int64_t> Explainer::appliedDelta() const {
  int64_t total = 0;
  bool complete = true;
  for (const UnitExplanation &U : units)
    for (const Alternative &A : U.alts)
      if (A.standing == Standing::Applied) {
        if (A.costDelta)
          total += *A.costDelta;
        else
          complete = false;
      }
  if (complete)
    return total;
  if (rec.selector == SolveSelector::DP && rec.selected)
    return *rec.selected;
  return std::nullopt;
}

// What the units cost as written, when every unit's cost is known.
std::optional<int64_t> Explainer::costBefore() const {
  int64_t sum = 0;
  for (const UnitExplanation &U : units) {
    if (!U.baseCost)
      return std::nullopt;
    sum += *U.baseCost;
  }
  return sum;
}

// The accuracy cost of the frontier point the solve selected.
std::optional<double> Explainer::selectedAcc() const {
  if (!rec.selected)
    return std::nullopt;
  auto it = rec.table.find(*rec.selected);
  if (it == rec.table.end())
    return std::nullopt;
  return it->second;
}

Error Explainer::build() {
  markSteps();
  for (size_t s = 0; s < st.subgraphs.size(); ++s)
    if (Error E = buildSubgraph(s))
      return E;
  // A unit over something other than a subgraph has an entry of its own.
  for (SiteUnit *unit : st.units) {
    if (unit->subgraphOf())
      continue;
    UnitExplanation U;
    U.id = (unit->info().letter + Twine(unit->indexInKind())).str();
    U.owner = unit;
    if (Error E = unit->explain(*this, U))
      return E;
    U.loc = locOf(U.root);
    judge(U);
    U.left = leftReason(U);
    units.push_back(std::move(U));
  }
  return Error::success();
}

// The explanation of a solved site, in the schema Explain.h documents.
json::Object Explainer::toJSON() {
  json::Object root;
  root["schema"] = kSchema;

  const SolvedFrom &from = st.get<SolvedFrom>();
  json::Object fn = functionJSON(Sess, F);
  fn["canonical_hash"] = from.canonicalHash;
  fn["profile"] = from.profilePath;
  root["function"] = std::move(fn);

  json::Object target = targetJSON(Sess.Opts, tau, rec.siteTarget);
  target["cost_model"] = costModelPath(Sess.get<CostModelState>());
  target["error_sum"] = errorSumName(st.errorSum);
  target["selector"] = selectorName(rec.selector);
  if (rec.fromCache)
    target["from_cache"] = true;
  if (!rec.sites.empty()) {
    json::Array sites;
    for (const std::string &name : rec.sites)
      sites.push_back(name);
    target["sites"] = std::move(sites);
  }
  root["target"] = std::move(target);

  json::Object scaleObj;
  putNum(scaleObj, "accuracy_scale", rec.accScale);
  putNum(scaleObj, "baseline_accuracy_cost", rec.baseline);
  putNum(scaleObj, "profiled_cost", rec.profiledCost);
  if (from.kappa) {
    scaleObj["kappa"] = *from.kappa;
    scaleObj["kappa_mark"] = kappaMarkName(from.kappaMark);
    scaleObj["kappa_factor"] = st.scaleFactor;
  }
  root["scale"] = std::move(scaleObj);

  json::Object result;
  std::optional<int64_t> before = costBefore();
  if (before)
    result["cost_before"] = scale.share(*before);
  std::optional<int64_t> delta = appliedDelta();
  if (before && delta)
    result["cost_after"] = scale.share(*before + *delta);
  putNum(result, "modelled_rel_error_as_written", relErr(0.0));
  double acc =
      selectedNotTaken()
          ? 0.0
          : selectedAcc().value_or(
                steps.empty() ? 0.0 : std::numeric_limits<double>::quiet_NaN());
  putNum(result, "modelled_rel_error", relErr(acc));
  json::Array appliedIds;
  json::Array stepList;
  for (const SolutionStep &step : steps) {
    const bool done = !skipped.count(keyOf(step));
    if (done)
      appliedIds.push_back(stepId(step));
    stepList.push_back(
        json::Object{{"id", stepId(step)},
                     {"choice", unitOf(step).choiceText(step.candidateIndex)},
                     {"applied", done}});
  }
  const bool rewritten = !appliedIds.empty();
  result["applied"] = std::move(appliedIds);
  result["steps"] = std::move(stepList);
  result["pin"] = pinValue(Sess, st, stepsToApply(steps));
  if (rec.selected) {
    json::Object selected;
    selected["cost"] = scale.share(*rec.selected);
    auto point = rec.table.find(*rec.selected);
    if (point != rec.table.end()) {
      putNum(selected, "accuracy_cost", point->second);
      putNum(selected, "modelled_rel_error", relErr(point->second));
    }
    result["selected"] = std::move(selected);
  }
  if (rec.time) {
    const SolveRecord::Time &t = *rec.time;
    json::Object time;
    putNum(time, "arithmetic_as_written", t.arithAsWritten);
    putNum(time, "arithmetic_selected", t.arithSelected);
    putNum(time, "memory", t.memory);
    putNum(time, "combined_as_written", t.combinedAsWritten);
    putNum(time, "combined_selected", t.combinedSelected);
    putNum(time, "pnorm", t.pnorm);
    putNum(time, "band", t.band);
    const double speedup = t.combinedAsWritten / t.combinedSelected;
    putNum(time, "modelled_speedup", speedup);
    time["inside_band"] = speedup > 1.0 && speedup <= 1.0 + t.band;
    time["taken"] = t.taken;
    result["time"] = std::move(time);
  }
  if (rewritten) {
    result["status"] = "rewritten";
  } else if (!rec.table.empty() && !rec.selected) {
    result["status"] = "no_solution";
    result["reason"] = siteLeftReason();
    if (tau > 0.0) {
      double best = std::numeric_limits<double>::infinity();
      for (const auto &[cost, acc] : rec.table)
        best = std::min(best, acc);
      putNum(result, "best_achievable_rel_error", relErr(best));
    }
  } else {
    result["status"] = "as_written";
    result["reason"] = siteLeftReason();
  }
  root["result"] = std::move(result);

  json::Array subgraphs;
  for (UnitExplanation &U : units) {
    json::Object u;
    u["id"] = U.id;
    u["kind"] = U.kind;
    putLoc(u, "loc", U.loc);
    if (U.baseCost)
      u["cost"] = scale.share(*U.baseCost);
    u["executions"] = (int64_t)U.executions;
    if (U.sg) {
      u["ops"] = std::move(U.ops);
      u["inputs"] = std::move(U.inputs);
      u["outputs"] = json::Array(U.outputs);
      json::Object original;
      if (U.outC.size() == 1) {
        if (auto *o = U.outputs[0].getAsObject())
          if (auto f = o->getString("fpcore"))
            original["fpcore"] = f->str();
        if (!U.outC[0].empty())
          original["c"] = U.outC[0];
      } else {
        json::Array cs, fs;
        bool allF = true;
        for (size_t k = 0; k < U.outputs.size(); ++k) {
          cs.push_back(U.outC[k]);
          auto *o = U.outputs[k].getAsObject();
          auto f = o ? o->getString("fpcore") : std::nullopt;
          if (f)
            fs.push_back(f->str());
          else
            allF = false;
        }
        original["c"] = std::move(cs);
        if (allF)
          original["fpcore"] = std::move(fs);
      }
      u["original"] = std::move(original);
    }
    mergeKeys(u, std::move(U.keys));
    if (!U.left.empty())
      u["left_as_written"] = U.left;
    json::Array alts;
    for (Alternative &A : U.alts) {
      json::Object a;
      if (!A.id.empty())
        a["id"] = A.id;
      a["kind"] = A.kind;
      a["summary"] = A.summary;
      if (!A.outputs.empty() && U.sg) {
        json::Array outs;
        for (unsigned o : A.outputs)
          outs.push_back((int64_t)o);
        a["outputs"] = std::move(outs);
      }
      if (A.c.size() == 1 && !A.c[0].empty())
        a["c"] = A.c[0];
      else if (A.c.size() > 1) {
        json::Array cs;
        for (const std::string &c : A.c)
          cs.push_back(c);
        a["c"] = std::move(cs);
      }
      if (A.costDelta)
        a["cost_delta"] = scale.share(*A.costDelta);
      if (A.err)
        a["modelled_error"] = *A.err;
      mergeKeys(a, std::move(A.keys));
      a["status"] = statusName(A.standing);
      if (!A.reason.empty())
        a["reason"] = A.reason;
      alts.push_back(std::move(a));
    }
    u["alternatives"] = std::move(alts);
    subgraphs.push_back(std::move(u));
  }
  root["subgraphs"] = std::move(subgraphs);

  if (!rec.table.empty()) {
    json::Array pareto;
    for (const auto &[cost, acc] : rec.table) {
      json::Object p;
      int64_t b = cost;
      const PointVerdict verdict = rec.verdicts.at(cost);
      if (verdict != PointVerdict::Unselectable)
        p["budget"] = scale.share(b);
      if (before)
        p["cost"] = scale.share(*before + b);
      putNum(p, "modelled_rel_error", relErr(acc));
      json::Array ids;
      auto it = rec.solutions.find(cost);
      if (it != rec.solutions.end())
        for (const SolutionStep &step : it->second) {
          std::string id = stepId(step);
          if (!id.empty())
            ids.push_back(id);
        }
      p["applied"] = std::move(ids);
      if (verdict == PointVerdict::Selected)
        p["selected"] = true;
      pareto.push_back(std::move(p));
    }
    root["pareto"] = std::move(pareto);
  }
  return root;
}

std::string Explainer::siteLeftReason() const {
  if (!rec.table.empty() && !rec.selected) {
    if (tau > 0.0)
      return "no frontier point meets the accuracy target " + remarkNumber(tau);
    return "no frontier point is within the cost budget " +
           costStr(scale.fromShare(Sess.Opts.Solver.CostBudget));
  }
  if (rec.selector == SolveSelector::Pinned)
    return "no -poseidon-pin item names a candidate of this site";
  if (selectedNotTaken())
    return kNotFaster.str();
  if (st.get<SolvedFrom>().kappaMark == KappaMark::Diverged)
    return "the profiling runs diverged under the smallest perturbation, so "
           "the site offers no candidate";
  return "the selected frontier point leaves the site as written";
}

void Explainer::emitRemarks() {
  OptimizationRemarkEmitter ORE(&F);
  DiagnosticLocation siteLoc = siteLocation(Sess, F);
  std::string siteName = llvm::demangle(siteSymbolOf(Sess, F));
  if (auto paren = siteName.find('('); paren != std::string::npos)
    siteName = siteName.substr(0, paren);

  std::optional<int64_t> before = costBefore();
  std::optional<int64_t> delta = appliedDelta();
  unsigned nApplied = 0;
  for (const UnitExplanation &U : units)
    for (const Alternative &A : U.alts)
      nApplied += A.standing == Standing::Applied;

  ORE.emit([&]() {
    OptimizationRemarkAnalysis R(kRemarkPass, "Site", siteLoc,
                                 &F.getEntryBlock());
    R << "Poseidon site " << ore::NV("Site", siteName) << ": ";
    if (tau > 0.0)
      R << "target " << ore::NV("Tau", remarkNumber(tau));
    else
      R << "cost budget "
        << ore::NV("Budget",
                   costStr(scale.fromShare(Sess.Opts.Solver.CostBudget)));
    double asWritten = relErr(0.0);
    if (std::isfinite(asWritten))
      R << "; modelled error as written "
        << ore::NV("ErrorBefore", remarkNumber(asWritten));
    if (nApplied == 0) {
      R << "; left as written";
    } else {
      double acc =
          selectedAcc().value_or(std::numeric_limits<double>::quiet_NaN());
      if (std::isfinite(relErr(acc)))
        R << ", after " << ore::NV("Error", remarkNumber(relErr(acc)));
      R << "; " << ore::NV("Rewrites", (unsigned)nApplied)
        << (nApplied == 1 ? " rewrite" : " rewrites");
      if (before && delta)
        R << ", cost "
          << ore::NV("Cost", costStr(*before) + " -> " +
                                 costStr(*before + *delta) +
                                 pctStr(*before, *before + *delta));
    }
    if (!rec.table.empty())
      R << "; " << ore::NV("ParetoPoints", (unsigned)rec.table.size())
        << (rec.table.size() == 1 ? " Pareto point" : " Pareto points");
    return R;
  });
  if (nApplied == 0)
    ORE.emit([&]() {
      OptimizationRemarkMissed R(kRemarkPass, "SiteLeftAsWritten", siteLoc,
                                 &F.getEntryBlock());
      R << "Poseidon left site " << ore::NV("Site", siteName)
        << " as written: " << ore::NV("Reason", siteLeftReason());
      return R;
    });

  for (const UnitExplanation &U : units) {
    if (!U.root)
      continue;
    ORE.emit([&]() {
      OptimizationRemarkAnalysis R(kRemarkPass, "Profile", U.root);
      if (!U.sg) {
        U.owner->remarkProfile(R, U);
        return R;
      }
      R << "Poseidon profile of " << ore::NV("Expr", exprWords(U));
      if (U.outC.size() > 1)
        R << " and " << ore::NV("MoreOutputs", (unsigned)U.outC.size() - 1)
          << " more output" << (U.outC.size() > 2 ? "s" : "");
      R << ": ";
      if (!U.inputRanges.empty())
        R << ore::NV("Ranges", join(U.inputRanges, ", ")) << "; ";
      R << ore::NV("Executions", U.executions) << " executions";
      if (std::isfinite(U.sensitivity))
        R << ", sensitivity "
          << ore::NV("Sensitivity", remarkNumber(U.sensitivity));
      return R;
    });
  }

  for (const UnitExplanation &U : units)
    for (const Alternative &A : U.alts) {
      if (A.standing != Standing::Applied || !A.at)
        continue;
      std::string costChange;
      if (A.baseCost && A.costDelta)
        costChange = costStr(*A.baseCost) + " -> " +
                     costStr(*A.baseCost + *A.costDelta) +
                     pctStr(*A.baseCost, *A.baseCost + *A.costDelta);
      std::string errChange;
      if (A.accDelta && std::isfinite(share(A.baseAcc)))
        errChange = remarkNumber(share(A.baseAcc)) + " -> " +
                    remarkNumber(share(A.baseAcc + *A.accDelta));
      A.unit->remarkApplied(ORE, *this, U, A, errChange, costChange);
    }

  for (const UnitExplanation &U : units) {
    if (!U.root)
      continue;
    if (!U.applied)
      ORE.emit([&]() {
        OptimizationRemarkMissed R(kRemarkPass, "LeftAsWritten", U.root);
        R << "Poseidon left " << ore::NV("Expr", U.sg ? exprWords(U) : U.name)
          << " as written: " << ore::NV("Reason", U.left);
        return R;
      });
    SmallVector<const Alternative *, 8> rejected;
    for (const Alternative &A : U.alts)
      if (A.standing != Standing::Applied)
        rejected.push_back(&A);
    std::stable_sort(rejected.begin(), rejected.end(),
                     [](const Alternative *a, const Alternative *b) {
                       if (a->standing != b->standing)
                         return a->standing < b->standing;
                       if (a->err && b->err)
                         return *a->err < *b->err;
                       return false;
                     });
    for (size_t k = 0; k < rejected.size() && k < kMaxRejectedRemarks; ++k) {
      const Alternative &A = *rejected[k];
      const Instruction *at = A.at ? A.at : U.root;
      ORE.emit([&]() {
        OptimizationRemarkMissed R(kRemarkPass, "Rejected", at);
        R << "Poseidon "
          << (A.standing == Standing::Pareto ? "did not select " : "rejected ")
          << ore::NV("Kind", A.kindWords);
        if (!A.id.empty())
          R << " " << ore::NV("Id", A.id);
        if (A.c.size() == 1 && !A.c[0].empty())
          R << " to " << ore::NV("Rewritten", clip(A.c[0]));
        else if (!A.expr)
          R << " (" << ore::NV("Rewritten", A.summary) << ")";
        else if (!A.expr->empty())
          R << " to " << ore::NV("Rewritten", clip(*A.expr));
        R << ": " << ore::NV("Reason", A.reason);
        SmallVector<std::string, 2> facts;
        if (A.err && A.reason.find("modelled error") == std::string::npos)
          facts.push_back("modelled error " + remarkNumber(*A.err));
        if (A.costDelta)
          facts.push_back("cost change " + costStr(*A.costDelta));
        if (!facts.empty())
          R << " (" << ore::NV("Facts", join(facts, ", ")) << ")";
        return R;
      });
    }
  }
}

bool explainWanted(const Options &O, const Function &F) {
  return !O.Diagnostics.Explain.empty() ||
         OptimizationRemarkEmitter::allowExtraAnalysis(F, kRemarkPass);
}

void explainLeftAlone(Session &S, Function &F, double tau, bool siteTarget,
                      StringRef status, StringRef reason, StringRef profile,
                      bool body) {
  if (S.Opts.Diagnostics.Explain.empty())
    return;
  json::Object result{{"status", status}, {"reason", reason}};
  if (!profile.empty())
    result["profile"] = profile;
  writeExplanation(S, F,
                   body ? bodyProfileStem(S.get<SiteRecords>(), F)
                        : siteFileStem(S.get<SiteRecords>(), F),
                   json::Object{{"schema", kSchema},
                                {"function", functionJSON(S, F)},
                                {"target", targetJSON(S.Opts, tau, siteTarget)},
                                {"result", std::move(result)}});
}

void explainNothingToOptimize(Session &S, Function &F, double tau,
                              bool siteTarget) {
  static constexpr StringLiteral reason =
      "Poseidon found no floating-point subgraph of two or more operations "
      "and no matrix product to optimize";
  OptimizationRemarkEmitter ORE(&F);
  ORE.emit([&]() {
    DiagnosticLocation loc = siteLocation(S, F);
    OptimizationRemarkMissed R(kRemarkPass, "NothingToOptimize", loc,
                               &F.getEntryBlock());
    R << reason;
    return R;
  });
  explainLeftAlone(S, F, tau, siteTarget, "nothing_to_optimize", reason);
}

void explainUnprofiled(Session &S, Function &F, double tau, bool siteTarget,
                       StringRef status, StringRef cause, StringRef profile) {
  OptimizationRemarkEmitter ORE(&F);
  ORE.emit([&]() {
    DiagnosticLocation loc = siteLocation(S, F);
    OptimizationRemarkMissed R(kRemarkPass, "Unprofiled", loc,
                               &F.getEntryBlock());
    R << "Poseidon left the site unchanged: " << ore::NV("Cause", cause);
    return R;
  });
  explainLeftAlone(S, F, tau, siteTarget, status, cause, profile);
}

Error explainSolve(Session &S, Function &F, SiteState &st,
                   ArrayRef<SolutionStep> steps, const SolveRecord &rec) {
  const bool remarks =
      OptimizationRemarkEmitter::allowExtraAnalysis(F, kRemarkPass);
  const bool write = !S.Opts.Diagnostics.Explain.empty();
  if (!remarks && !write)
    return Error::success();
  Explainer E(S, F, st, steps, rec);
  if (Error Err = E.build())
    return Err;
  if (remarks)
    E.emitRemarks();
  if (write)
    writeExplanation(S, F, siteFileStem(S.get<SiteRecords>(), F), E.toJSON());
  return Error::success();
}

} // namespace poseidon
