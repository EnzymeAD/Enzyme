//===- Explain.cpp - what a solve changed, considered and rejected --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Explain.h"
#include "Model/CostModel.h"
#include "Model/FPCore.h"
#include "Optimize.h"
#include "Pass/Diagnostics.h"
#include "Pass/Session.h"
#include "Sites/SiteIdentity.h"
#include "SolveSites.h"
#include "Support/CacheKey.h"
#include "Support/Utils.h"
#include "Targets/MathLibrary.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision/Precision.h"
#include "Techniques/Rewrite/Rewrite.h"
#include "runtime/profile/ProfileFormat.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
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

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

namespace poseidon {

static constexpr size_t kMaxRejectedRemarks = 3;
static constexpr size_t kRemarkExprChars = 160;
static constexpr size_t kMaxListedOps = 6;

static void putNum(json::Object &o, StringRef key, double v) {
  if (std::isfinite(v))
    o[key] = v;
}

static std::string num(double v, int digits = 2) {
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

static std::string clip(const std::string &s, size_t n = kRemarkExprChars) {
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

static std::string precisionWords(PrecisionChangeType t) {
  if (unsigned k = expansionComponents(t))
    return formatv("a {0}-component float expansion", k).str();
  switch (t) {
  case PrecisionChangeType::BF16:
    return "bfloat16";
  case PrecisionChangeType::FP16:
    return "half";
  case PrecisionChangeType::FP32:
    return "float";
  case PrecisionChangeType::FP64:
    return "double";
  case PrecisionChangeType::FP80:
    return "long double";
  case PrecisionChangeType::FP128:
    return "quad";
  default:
    return formatOf(t).Name.str();
  }
}

static std::string opWord(StringRef op) {
  if (op == "+" || op == "-" || op == "*" || op == "/")
    return ("'" + op + "'").str();
  if (op == "neg")
    return "negation";
  return op.str();
}

static std::string kindWords(StringRef kind) {
  if (kind == "algebraic")
    return "the rewrite";
  if (kind == "precision")
    return "the precision change";
  if (kind == "expansion")
    return "the float expansion";
  return "the product candidate";
}

static std::string joinWords(ArrayRef<std::string> items) {
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

static bool isPolynomial(StringRef expr) {
  static const StringSet<> arith = {"+",   "-",   "*",       "/",     "neg",
                                    "fma", "pow", "literal", "approx"};
  bool head = false;
  size_t i = 0;
  while (i < expr.size()) {
    char c = expr[i];
    if (c == '(' || expr.substr(i).starts_with("#s(")) {
      i += c == '(' ? 1 : 3;
      head = true;
      continue;
    }
    if (c == ')' || std::isspace((unsigned char)c)) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < expr.size() && !std::isspace((unsigned char)expr[j]) &&
           expr[j] != '(' && expr[j] != ')')
      ++j;
    if (head && !arith.contains(expr.substr(i, j - i).split('.').first))
      return false;
    head = false;
    i = j;
  }
  return true;
}

static const Instruction *
firstInIROrder(Function &F, const SmallPtrSetImpl<const Value *> &set) {
  for (Instruction &I : instructions(F))
    if (set.contains(&I))
      return &I;
  return nullptr;
}

namespace {

struct Alt {
  std::string id, kind, summary, fpcore, precision, status, reason;
  std::string inKernelClass;
  SmallVector<std::string, 1> c;
  SmallVector<unsigned, 1> outputs;
  bool approx = false, series = false;
  std::string approxNote;
  SmallVector<std::string, 2> approxRanges;
  std::optional<int64_t> costDelta;
  std::optional<double> accDelta, err, domainError;
  std::optional<double> bitsBefore, bitsAfter;
  json::Array changes;
  SmallVector<std::string, 4> changedOps;
  std::string changedTo;
  const Instruction *at = nullptr;
  const SiteUnit *item = nullptr;
  size_t index = 0;
  int rank = 9;
  double baseAcc = std::numeric_limits<double>::quiet_NaN();
  std::optional<int64_t> baseCost;
};

struct Unit {
  std::string id, kind;
  const Subgraph *sg = nullptr;
  SourceLoc loc;
  const Instruction *root = nullptr;
  json::Array ops, inputs, outputs, herbie;
  SmallVector<std::string, 2> outC;
  SmallVector<std::string, 2> inputRanges;
  std::vector<Alt> alts;
  std::optional<int64_t> baseCost;
  uint64_t executions = 0;
  double sensitivity = std::numeric_limits<double>::quiet_NaN();
  std::string left, shape;
  bool applied = false;
};

} // namespace

static std::string exprWords(const Unit &U) {
  if (!U.outC.empty() && !U.outC[0].empty())
    return clip(U.outC[0]);
  return formatv("the subgraph of {0} operations at {1}:{2}", U.ops.size(),
                 U.loc.line, U.loc.col)
      .str();
}

static std::string siteSymbolOf(const Session &S, const Function &F) {
  return siteSymbol(siteKey(S.Sites, F)).str();
}

// The schema identifier of every explanation; see Explainer::toJSON.
static constexpr StringLiteral kSchema = "poseidon-explain/1";

// The names of the site whose clone is F, or whose body is F when no clone was
// made, where it is called and where it is written.
static json::Object functionJSON(const Session &S, const Function &F) {
  json::Object fn;
  const Function *origin = siteOriginOf(S.Sites, F);
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

// The directory of -poseidon-explain.
static const std::string &explanationDir(const Options &O) {
  return O.Diagnostics.Explain;
}

// Writes `doc`, the explanation of a site, as <stem>.explain.json into the
// explanation directory.
static void writeExplanation(const Session &S, const Function &F,
                             StringRef stem, json::Object doc) {
  const std::string &dir = explanationDir(S.Opts);
  if (auto EC = sys::fs::create_directories(dir)) {
    diagnose(F, F.getSubprogram(),
             "cannot create the explanation directory " + dir + ": " +
                 EC.message(),
             DS_Warning);
    return;
  }
  SmallString<128> path(dir);
  sys::path::append(path, stem + ".explain.json");
  if (Error Err = writeFileAtomically(
          path, formatv("{0:2}", json::Value(std::move(doc))).str() + "\n")) {
    diagnose(F, F.getSubprogram(),
             "cannot write the explanation " + path + ": " +
                 toString(std::move(Err)),
             DS_Warning);
    return;
  }
  if (S.Opts.Diagnostics.Print)
    errs() << "[poseidon] explanation written to " << path << '\n';
}

// Where a remark about a site points: its call, else the site's function.
static DiagnosticLocation siteLocation(const Session &S, const Function &F) {
  auto site = S.SiteCalls.find(&F);
  return site != S.SiteCalls.end() && site->second
             ? DiagnosticLocation(site->second)
             : DiagnosticLocation(F.getSubprogram());
}

namespace {

class Explainer {
  Session &Sess;
  Function &F;
  SiteState &st;
  ArrayRef<SolutionStep> steps;
  const SolveRecord &rec;
  ValueNamer namer;
  // An overlay: fpcoreToC may add to the graph it is given, and the
  // explanation must leave the solve's own as it is.
  FPGraph scratch;
  std::set<std::pair<const SiteUnit *, size_t>> applied, skipped, onPareto,
      inSelected;
  std::map<std::pair<const SiteUnit *, size_t>, int64_t> firstBudget;
  std::vector<Unit> units;
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
    return (A0 + acc) / S;
  }

  double share(double acc) const {
    if (!(S > 0.0) || !std::isfinite(acc))
      return std::numeric_limits<double>::quiet_NaN();
    return acc / S;
  }

  FPNode *nodeOf(Value *V) const {
    auto it = st.valueToNodeMap.find(V);
    return it == st.valueToNodeMap.end() ? nullptr : it->second;
  }

  CRendering rendering(const Subgraph &sg) {
    CRendering how;
    how.inputs = &sg.inputs;
    how.leafName = [this](Value *V) { return namer.name(V); };
    return how;
  }

  std::string rangeOf(Value *V, const std::string &label) {
    auto it = st.valueToNodeMap.find(V);
    if (it == st.valueToNodeMap.end() || !it->second)
      return "";
    double lo = it->second->getLowerBound(), hi = it->second->getUpperBound();
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
      return "";
    int digits = 3;
    while (lo != hi && digits < 17 && num(lo, digits) == num(hi, digits))
      ++digits;
    return label + " in [" + num(lo, digits) + ", " + num(hi, digits) + "]";
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
  Error buildMatmul(size_t m);
  Error addCOAlternatives(Unit &U, const Subgraph &sg);
  Error addCSAlternatives(Unit &U, const Subgraph &sg);
  // `cost` on the solve's axis; an error at the site when the axis cannot hold
  // it.
  Expected<int64_t> onAxis(double cost) const;
  void judge(Unit &U);
  std::string leftReason(const Unit &U, const Subgraph *sg);
  std::optional<int64_t> appliedDelta() const;
  std::optional<int64_t> costBefore() const;
  std::optional<double> selectedAcc() const;
  // Why the solve left the site as written, when it applied no step.
  std::string siteLeftReason() const;

public:
  Explainer(Session &Sess, Function &F, SiteState &st,
            ArrayRef<SolutionStep> steps, const SolveRecord &rec)
      : Sess(Sess), F(F), st(st), steps(steps), rec(rec), namer(F),
        scratch(st.overlay()), A0(rec.baseline), S(rec.accScale), tau(rec.tau),
        scale(rec.profiledCost) {}

  Error build();
  void emitRemarks();
  json::Object toJSON();
};

} // namespace

void Explainer::markSteps() {
  for (const SolutionStep &step : stepsToApply(steps))
    applied.insert(keyOf(step));
  for (const SolutionStep &step : steps)
    if (!applied.count(keyOf(step)))
      skipped.insert(keyOf(step));
  for (const auto &[cost, sol] : rec.solutions)
    for (const SolutionStep &step : sol) {
      onPareto.insert(keyOf(step));
      firstBudget.try_emplace(keyOf(step), cost);
    }
  if (rec.selected) {
    auto it = rec.solutions.find(*rec.selected);
    if (it != rec.solutions.end())
      for (const SolutionStep &step : it->second)
        inSelected.insert(keyOf(step));
  }
}

Expected<int64_t> Explainer::onAxis(double cost) const {
  if (std::optional<DPCost> c = scale.toDPCost(cost))
    return *c;
  std::string Text;
  raw_string_ostream(Text) << cost;
  return userError(F, F.getSubprogram(),
                   "a cost of " + Twine(Text) +
                       " does not fit the DP's cost axis; a cost-model row "
                       "prices a candidate beyond 2^30 times the profiled "
                       "cost");
}

Error Explainer::addCOAlternatives(Unit &U, const Subgraph &sg) {
  CRendering how = rendering(sg);
  for (size_t i = 0; i < st.rewriteUnits.size(); ++i) {
    RewriteUnit &RU = st.rewriteUnits[i];
    if (RU.subgraph != &sg)
      continue;
    json::Object core;
    core["id"] = formatv("R{0}", i).str();
    if (!RU.herbiePrecision.empty())
      core["precision"] = RU.herbiePrecision;
    if (!RU.herbieStatus.empty())
      core["status"] = RU.herbieStatus;
    core["candidates"] = (int64_t)RU.candidates.size();
    if (!RU.unpricedReason.empty())
      core["unpriced"] = RU.unpricedReason;
    if (std::isfinite(RU.herbieBits) && std::isfinite(RU.initialHerbieAccuracy))
      core["error_bits"] = (1.0 - RU.initialHerbieAccuracy) * RU.herbieBits;
    U.herbie.push_back(std::move(core));

    SmallVector<unsigned, 1> outs;
    for (Value *root : RU.roots)
      for (unsigned k = 0; k < sg.outputs.size(); ++k)
        if (sg.outputs[k] == root)
          outs.push_back(k);
    std::string rdt = dtypeOfType(RU.roots.front()->getType());
    std::optional<int64_t> base;
    if (std::isfinite(RU.initialCompCost)) {
      Expected<int64_t> c = onAxis(RU.initialCompCost * (double)RU.executions);
      if (!c)
        return c.takeError();
      base = *c;
    }

    auto fill = [&](Alt &A, const std::string &expr) {
      A.kind = "algebraic";
      A.precision = RU.herbiePrecision;
      A.fpcore = expr;
      A.outputs = outs;
      A.at = cast<Instruction>(RU.roots.front());
      A.baseAcc = RU.initialAccCost;
      A.baseCost = base;
      SmallVector<std::pair<std::string, std::string>, 2> specs;
      A.c = fpcoreToC(expr, scratch, how, rdt, &specs);
      SmallVector<std::string, 2> notes;
      std::set<std::string> seen;
      for (const auto &[spec, impl] : specs) {
        if (!seen.insert(spec).second)
          continue;
        SmallVector<std::string, 1> specC = fpcoreToC(spec, scratch, how, "");
        std::string specText = clip(specC.empty() ? spec : specC[0], 120);
        if (!isPolynomial(impl)) {
          SmallVector<std::string, 1> implC = fpcoreToC(impl, scratch, how, "");
          notes.push_back(clip(implC.empty() ? impl : implC[0], 120) +
                          " in place of " + specText);
          continue;
        }
        A.series = true;
        notes.push_back("series approximation of " + specText);
        SmallSet<std::string, 8> args;
        getUniqueArgs(spec, args);
        for (const std::string &arg : args) {
          auto sv = scratch.symbolToValueMap.find(arg);
          if (sv == scratch.symbolToValueMap.end() || !sv->second)
            continue;
          std::string label = leafLabel(sv->second);
          std::string r = rangeOf(sv->second, label.empty() ? arg : label);
          if (!r.empty() && !llvm::is_contained(A.approxRanges, r))
            A.approxRanges.push_back(r);
        }
      }
      A.approx = !specs.empty();
      if (StringRef(expr).contains("(if"))
        notes.push_back("regime split");
      A.approxNote = joinWords(notes);
      std::string what = "Herbie rewrite";
      if (!A.precision.empty())
        what += " searched at " + A.precision;
      if (A.series)
        what += ", series approximation";
      else if (A.approx)
        what += ", approximation";
      if (StringRef(expr).contains("(if"))
        what += ", regime split";
      A.summary = what;
    };

    for (size_t j = 0; j < RU.candidates.size(); ++j) {
      const RewriteCandidate &cand = RU.candidates[j];
      Alt A;
      A.id = formatv("R{0}_{1}", i, cand.ordinal).str();
      A.item = &RU;
      A.index = j;
      fill(A, cand.expr);
      if (cand.CompCost != std::numeric_limits<double>::max()) {
        Expected<int64_t> c = onAxis(RU.getCompCostDelta(j));
        if (!c)
          return c.takeError();
        A.costDelta = *c;
      }
      double acc = RU.getAccCostDelta(j);
      if (std::isfinite(acc)) {
        A.accDelta = acc;
        double e = relErr(acc);
        if (std::isfinite(e))
          A.err = e;
      }
      if (std::isfinite(RU.herbieBits)) {
        if (std::isfinite(RU.initialHerbieAccuracy))
          A.bitsBefore = (1.0 - RU.initialHerbieAccuracy) * RU.herbieBits;
        if (std::isfinite(cand.herbieAccuracy))
          A.bitsAfter = (1.0 - cand.herbieAccuracy) * RU.herbieBits;
      }
      U.alts.push_back(std::move(A));
    }
    for (const auto &[expr, why] : RU.dropped) {
      Alt A;
      fill(A, expr);
      A.status = "rejected";
      A.reason = "the accuracy model dropped it: " + why;
      A.rank = 4;
      U.alts.push_back(std::move(A));
    }
  }
  return Error::success();
}

Error Explainer::addCSAlternatives(Unit &U, const Subgraph &sg) {
  CRendering how = rendering(sg);
  DenseMap<const FPNode *, unsigned> opIndex;
  unsigned k = 0;
  for (Instruction &I : instructions(F))
    if (sg.operations.contains(&I)) {
      if (FPNode *N = nodeOf(&I))
        opIndex[N] = k;
      ++k;
    }

  for (size_t i = 0; i < st.precisionUnits.size(); ++i) {
    PrecisionUnit &PU = st.precisionUnits[i];
    if (PU.subgraph != &sg)
      continue;
    auto describe = [&](Alt &A, PrecisionCandidate &cand) {
      bool expansion = false;
      DenseMap<const FPNode *, std::string> dtypes;
      SmallPtrSet<const Value *, 8> changedInsts;
      SmallVector<std::string, 4> parts;
      for (const PrecisionChange &ch : cand.changes) {
        expansion |= expansionComponents(ch.newType) != 0;
        json::Object c;
        c["from"] = formatOf(ch.oldType).Name.str();
        c["to"] = formatOf(ch.newType).Name.str();
        json::Array idx;
        SmallVector<std::string, 4> names;
        MapVector<std::string, unsigned, std::map<std::string, unsigned>>
            counts;
        for (FPLLValue *node : ch.nodes) {
          auto it = opIndex.find(node);
          if (it != opIndex.end())
            idx.push_back((int64_t)it->second);
          changedInsts.insert(node->value);
          SourceLoc L = locOf(node->value);
          std::string w = opWord(operationName(*node));
          ++counts[w];
          if (L.valid())
            w += formatv(" ({0}:{1})", L.line, L.col).str();
          names.push_back(w);
          A.changedOps.push_back(w);
          if (!expansionComponents(ch.newType))
            dtypes[node] =
                dtypeOfType(getLLVMFPType(ch.newType, F.getContext()));
        }
        c["ops"] = std::move(idx);
        A.changes.push_back(std::move(c));
        if (names.size() > kMaxListedOps) {
          names.clear();
          for (const auto &[w, n] : counts)
            names.push_back(n == 1 ? w : formatv("{0} x {1}", n, w).str());
        }
        parts.push_back(joinWords(names) + " in " + precisionWords(ch.newType));
        A.changedTo = precisionWords(ch.newType);
      }
      A.kind = expansion ? "expansion" : "precision";
      A.summary = cand.desc + ": " + joinWords(parts) + " (" +
                  std::to_string(A.changedOps.size()) + " of " +
                  std::to_string(sg.operations.size()) + " operations)";
      if (A.changedOps.size() == sg.operations.size() &&
          cand.changes.size() == 1)
        A.changedOps = {
            formatv("all {0} operations", sg.operations.size()).str()};
      else if (A.changedOps.size() > kMaxListedOps)
        A.changedOps = {formatv("{0} of the {1} operations",
                                A.changedOps.size(), sg.operations.size())
                            .str()};
      A.at = firstInIROrder(F, changedInsts);
      if (!A.at)
        A.at = sg.outputs[0];
      if (!expansion) {
        CRendering typed = how;
        typed.dtypes = &dtypes;
        for (Instruction *O : sg.outputs)
          if (FPNode *N = nodeOf(O))
            A.c.push_back(renderC(*N, typed, dtypeOfType(O->getType())));
      }
      A.baseAcc = PU.initialAccCost;
      A.baseCost = U.baseCost;
    };
    for (size_t j = 0; j < PU.candidates.size(); ++j) {
      PrecisionCandidate &cand = PU.candidates[j];
      Alt A;
      A.id = formatv("P{0}_{1}", i, cand.ordinal).str();
      A.item = &PU;
      A.index = j;
      describe(A, cand);
      for (unsigned o = 0; o < sg.outputs.size(); ++o)
        A.outputs.push_back(o);
      std::optional<int64_t> cost;
      if (cand.CompCost != std::numeric_limits<double>::max()) {
        Expected<int64_t> c = onAxis(PU.getCompCostDelta(j));
        if (!c)
          return c.takeError();
        cost = *c;
      }
      double acc = PU.getAccCostDelta(j);
      PrecisionUnit::RewriteUnitSet cos;
      for (const auto &[item, idx] : applied)
        for (RewriteUnit &RU : st.rewriteUnits)
          if (&RU == item && RU.subgraph == &sg)
            cos.insert(&RU);
      if (applied.count({&PU, j}) && !cos.empty()) {
        PrecisionUnit::RepriceKey key{j, cos};
        auto c = PU.compCostDeltaCache.find(key);
        cost = std::nullopt;
        if (c != PU.compCostDeltaCache.end()) {
          Expected<int64_t> repriced = onAxis(c->second);
          if (!repriced)
            return repriced.takeError();
          cost = *repriced;
        }
        auto a = PU.accCostDeltaCache.find(key);
        acc = a == PU.accCostDeltaCache.end()
                  ? std::numeric_limits<double>::quiet_NaN()
                  : a->second;
      }
      A.costDelta = cost;
      if (std::isfinite(acc)) {
        A.accDelta = acc;
        double e = relErr(acc);
        if (std::isfinite(e))
          A.err = e;
      }
      U.alts.push_back(std::move(A));
    }
    for (const auto &[desc, why] : PU.dropped) {
      Alt A;
      A.kind = "precision";
      A.summary = desc;
      A.at = sg.outputs[0];
      A.status = "rejected";
      A.reason = "pricing dropped it: " + why;
      A.rank = 4;
      U.alts.push_back(std::move(A));
    }
  }
  return Error::success();
}

void Explainer::judge(Unit &U) {
  for (Alt &A : U.alts) {
    if (!A.item)
      continue;
    auto key = std::make_pair(A.item, A.index);
    if (applied.count(key)) {
      A.status = "applied";
      A.rank = -1;
      U.applied = true;
      if (rec.selector == SolveSelector::Pinned)
        A.reason = "selected by -poseidon-pin";
      else if (tau > 0.0)
        A.reason = "part of the cheapest solution that meets the target";
      else
        A.reason = "part of the most accurate solution within the cost budget";
      continue;
    }
    A.status = "rejected";
    if (skipped.count(key)) {
      A.reason = "another selected rewrite of the same output was applied "
                 "first";
      A.rank = 3;
      continue;
    }
    if (inSelected.count(key)) {
      A.reason = "it edits instructions a selected matrix-product raise "
                 "consumes";
      A.rank = 3;
      continue;
    }
    if (tau > 0.0 && A.err && *A.err > tau) {
      A.reason =
          "modelled error " + num(*A.err) + " is above the target " + num(tau);
      A.rank = A.costDelta && *A.costDelta < 0 ? 0 : 2;
      continue;
    }
    if (A.costDelta && A.accDelta && *A.costDelta >= 0 && *A.accDelta >= 0) {
      A.reason = "neither cheaper nor more accurate than as written";
      A.rank = 3;
      continue;
    }
    if (onPareto.count(key)) {
      A.status = "pareto";
      A.reason = "on the Pareto frontier";
      auto b = firstBudget.find(key);
      auto sol = b == firstBudget.end() ? rec.solutions.end()
                                        : rec.solutions.find(b->second);
      if (sol != rec.solutions.end()) {
        std::string ids;
        for (const SolutionStep &step : sol->second)
          ids += (ids.empty() ? "" : ",") + stepId(step);
        A.reason += "; -poseidon-pin=" + ids + " applies it";
      }
      A.rank = 1;
      continue;
    }
    const Alt *dom = nullptr;
    auto better = [&](const Alt &B, const Alt &D) {
      bool b = applied.count({B.item, B.index}),
           d = applied.count({D.item, D.index});
      return b != d ? b : *B.costDelta < *D.costDelta;
    };
    if (A.costDelta && A.accDelta)
      for (const Alt &B : U.alts)
        if (&B != &A && B.item && B.costDelta && B.accDelta &&
            *B.costDelta <= *A.costDelta && *B.accDelta <= *A.accDelta &&
            (*B.costDelta < *A.costDelta || *B.accDelta < *A.accDelta) &&
            (!dom || better(B, *dom)))
          dom = &B;
    if (dom) {
      A.reason = "dominated by " + dom->id +
                 ", which costs no more and is at least as accurate";
      A.rank = 3;
      continue;
    }
    const Alt *rival = nullptr;
    for (const Alt &B : U.alts)
      if (&B != &A && B.item && applied.count({B.item, B.index}) &&
          B.kind == A.kind && B.outputs == A.outputs)
        rival = &B;
    if (rival) {
      A.reason = A.kind == "algebraic"
                     ? "the selected " + rival->id + " rewrites the same output"
                     : "the subgraph takes one precision assignment, and " +
                           rival->id + " was selected";
      A.rank = 3;
      continue;
    }
    if (tau == 0.0 && A.costDelta &&
        *A.costDelta > scale.fromShare(Sess.Opts.Solver.CostBudget)) {
      A.reason = "its cost change " + costStr(*A.costDelta) +
                 " is above the budget " +
                 costStr(scale.fromShare(Sess.Opts.Solver.CostBudget));
      A.rank = 2;
      continue;
    }
    A.reason = "not on the Pareto frontier of the site";
    A.rank = A.accDelta && *A.accDelta < 0 ? 2 : 3;
  }
}

std::string Explainer::leftReason(const Unit &U, const Subgraph *sg) {
  if (U.applied)
    return "";
  bool any = llvm::any_of(U.alts, [](const Alt &A) { return A.item; });
  if (!any) {
    SmallVector<std::string, 2> why;
    if (sg) {
      bool hadCO = false;
      for (const RewriteUnit &RU : st.rewriteUnits)
        if (RU.subgraph == sg) {
          hadCO = true;
          if (!RU.unpricedReason.empty())
            why.push_back("its candidates could not be priced: " +
                          RU.unpricedReason);
          else if (!RU.herbieStatus.empty() && RU.candidates.empty())
            why.push_back("Herbie returned no alternative (" + RU.herbieStatus +
                          ")");
        }
      for (const PrecisionUnit &PU : st.precisionUnits)
        if (PU.subgraph == sg && !PU.unpricedReason.empty())
          why.push_back("its candidates could not be priced: " +
                        PU.unpricedReason);
      if (!hadCO) {
        if (!Sess.Opts.Candidates.EnableRewrites)
          why.push_back("the algebraic search is disabled");
        else
          for (Instruction *O : sg->outputs) {
            auto it = st.valueToNodeMap.find(O);
            if (it != st.valueToNodeMap.end() && it->second &&
                it->second->grad == 0.0) {
              why.push_back("the profiled gradient of its output is zero");
              break;
            }
          }
      }
    }
    if (why.empty())
      why.push_back("no candidate was generated");
    llvm::sort(why);
    why.erase(std::unique(why.begin(), why.end()), why.end());
    return joinWords(why);
  }
  if (tau > 0.0) {
    const Alt *best = nullptr;
    for (const Alt &A : U.alts)
      if (A.item && A.err && (!best || *A.err < *best->err))
        best = &A;
    bool anyMeets = llvm::any_of(
        U.alts, [&](const Alt &A) { return A.item && A.err && *A.err <= tau; });
    if (!anyMeets && best)
      return "every alternative's modelled error is above the target " +
             num(tau) + " (best: " + best->id + ", " + num(*best->err) + ")";
    bool cheaperMeets = llvm::any_of(U.alts, [&](const Alt &A) {
      return A.item && A.err && *A.err <= tau && A.costDelta &&
             *A.costDelta < 0;
    });
    if (!cheaperMeets)
      return "no alternative that meets the target " + num(tau) +
             " is cheaper than as written";
    if (!rec.selected && steps.empty())
      return "no solution on the site's frontier meets the target " + num(tau);
    return "its alternatives do not fit the target together with the "
           "selected rewrites";
  }
  return "no alternative lowers the modelled error within the cost budget " +
         costStr(scale.fromShare(Sess.Opts.Solver.CostBudget));
}

Error Explainer::buildSubgraph(size_t s) {
  Subgraph &sg = st.subgraphs[s];
  Unit U;
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
      double lo = N->getLowerBound(), hi = N->getUpperBound();
      if (std::isfinite(lo) && std::isfinite(hi) && lo <= hi) {
        op["min"] = lo;
        op["max"] = hi;
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
      double lo = N->getLowerBound(), hi = N->getUpperBound();
      if (std::isfinite(lo) && std::isfinite(hi) && lo <= hi) {
        in["min"] = lo;
        in["max"] = hi;
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
    for (const RewriteUnit &RU : st.rewriteUnits)
      if (RU.subgraph == &sg && !RU.isArray() && RU.roots.front() == O) {
        out["fpcore"] = RU.expr;
        break;
      }
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
    Expected<double> base = compCost(Sess.CostModel, outs, sg.inputs);
    if (!base)
      return base.takeError();
    if (FPNode *N = nodeOf(sg.outputs[0])) {
      Expected<int64_t> c = onAxis(*base * (double)N->executions);
      if (!c)
        return c.takeError();
      U.baseCost = *c;
    }
  }

  if (Error E = addCOAlternatives(U, sg))
    return E;
  if (Error E = addCSAlternatives(U, sg))
    return E;
  judge(U);
  U.left = leftReason(U, &sg);
  units.push_back(std::move(U));
  return Error::success();
}

Error Explainer::buildMatmul(size_t m) {
  MatmulUnit &MU = st.matmulUnits[m];
  Unit U;
  U.id = formatv("M{0}", m).str();
  U.kind = "matmul";
  const RecognizedMatmul *mm = MU.matmul;
  if (mm) {
    U.root = dyn_cast_or_null<Instruction>(mm->outputValue);
    if (!U.root && !mm->footprint.empty())
      U.root = mm->footprint[0];
    auto dim = [](unsigned d) {
      return d == ~0u || d == 0 ? std::string("?") : std::to_string(d);
    };
    unsigned M = mm->globalM ? mm->globalM : mm->M;
    unsigned N = mm->globalN ? mm->globalN : mm->N;
    unsigned K = mm->globalK ? mm->globalK : mm->K;
    U.shape = dim(M) + "x" + dim(N) + "x" + dim(K);
  }
  U.loc = locOf(U.root);
  U.executions = MU.executions;
  if (MU.initialCompCost >= 0.0) {
    Expected<int64_t> c = onAxis(MU.initialCompCost * (double)MU.executions);
    if (!c)
      return c.takeError();
    U.baseCost = *c;
  }
  for (size_t j = 0; j < MU.candidates.size(); ++j) {
    const MatmulUnit::Option &o = MU.candidates[j];
    Alt A;
    A.id = formatv("M{0}_{1}", m, j).str();
    A.kind = "matmul";
    A.summary = matmulOptionLabel(o);
    if (o.realization == Realization::InKernel)
      A.inKernelClass = inKernelCostClass(o.variant);
    A.item = &MU;
    A.index = j;
    A.at = U.root;
    Expected<int64_t> c = onAxis(MU.getCompCostDelta(j));
    if (!c)
      return c.takeError();
    A.costDelta = *c;
    double acc = MU.getAccCostDelta(j);
    if (std::isfinite(acc)) {
      A.accDelta = acc;
      if (std::isfinite(relErr(acc)))
        A.err = relErr(acc);
    }
    if (std::isfinite(o.domainError))
      A.domainError = o.domainError;
    A.baseCost = U.baseCost;
    U.alts.push_back(std::move(A));
  }
  judge(U);
  U.left = leftReason(U, nullptr);
  units.push_back(std::move(U));
  return Error::success();
}

std::optional<int64_t> Explainer::appliedDelta() const {
  int64_t total = 0;
  bool complete = true;
  for (const Unit &U : units)
    for (const Alt &A : U.alts)
      if (A.status == "applied") {
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
  for (const Unit &U : units) {
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
  for (size_t m = 0; m < st.matmulUnits.size(); ++m)
    if (Error E = buildMatmul(m))
      return E;
  return Error::success();
}

// The explanation of a solved site, in the schema Explain.h documents.
json::Object Explainer::toJSON() {
  json::Object root;
  root["schema"] = kSchema;

  json::Object fn = functionJSON(Sess, F);
  fn["canonical_hash"] = st.canonicalHash;
  fn["profile"] = st.profilePath;
  root["function"] = std::move(fn);

  json::Object target = targetJSON(Sess.Opts, tau, rec.siteTarget);
  target["cost_model"] = costModelPath(Sess.CostModel);
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
  if (st.kappa) {
    scaleObj["kappa"] = *st.kappa;
    scaleObj["kappa_mark"] = kappaMarkName(st.kappaMark);
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
  double acc = selectedAcc().value_or(
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
  for (Unit &U : units) {
    json::Object u;
    u["id"] = U.id;
    u["kind"] = U.kind;
    putLoc(u, "loc", U.loc);
    if (!U.shape.empty())
      u["shape"] = U.shape;
    if (U.baseCost)
      u["cost"] = scale.share(*U.baseCost);
    u["executions"] = (int64_t)U.executions;
    if (U.kind == "subgraph") {
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
      if (!U.herbie.empty())
        u["herbie"] = std::move(U.herbie);
    }
    if (!U.left.empty())
      u["left_as_written"] = U.left;
    json::Array alts;
    for (Alt &A : U.alts) {
      json::Object a;
      if (!A.id.empty())
        a["id"] = A.id;
      a["kind"] = A.kind;
      a["summary"] = A.summary;
      if (!A.inKernelClass.empty())
        a["inkernel_class"] = A.inKernelClass;
      if (!A.precision.empty())
        a["precision"] = A.precision;
      if (!A.outputs.empty() && U.kind == "subgraph") {
        json::Array outs;
        for (unsigned o : A.outputs)
          outs.push_back((int64_t)o);
        a["outputs"] = std::move(outs);
      }
      if (!A.fpcore.empty())
        a["fpcore"] = A.fpcore;
      if (A.c.size() == 1 && !A.c[0].empty())
        a["c"] = A.c[0];
      else if (A.c.size() > 1) {
        json::Array cs;
        for (const std::string &c : A.c)
          cs.push_back(c);
        a["c"] = std::move(cs);
      }
      if (A.kind == "algebraic") {
        a["approx"] = A.approx;
        if (!A.approxNote.empty())
          a["approx_note"] = A.approxNote;
      }
      if (!A.changes.empty())
        a["changes"] = std::move(A.changes);
      if (A.costDelta)
        a["cost_delta"] = scale.share(*A.costDelta);
      if (A.err)
        a["modelled_error"] = *A.err;
      if (A.domainError)
        a["modelled_domain_error"] = *A.domainError;
      if (A.bitsBefore || A.bitsAfter) {
        json::Object bits;
        if (A.bitsBefore)
          bits["before"] = *A.bitsBefore;
        if (A.bitsAfter)
          bits["after"] = *A.bitsAfter;
        a["herbie_error_bits"] = std::move(bits);
      }
      a["status"] = A.status;
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
      if (selectable(Sess.Opts, acc))
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
      if (rec.selected && *rec.selected == b)
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
      return "no frontier point meets the accuracy target " + num(tau);
    return "no frontier point is within the cost budget " +
           costStr(scale.fromShare(Sess.Opts.Solver.CostBudget));
  }
  if (rec.selector == SolveSelector::Pinned)
    return "no -poseidon-pin item names a candidate of this site";
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
  for (const Unit &U : units)
    for (const Alt &A : U.alts)
      nApplied += A.status == "applied";

  ORE.emit([&]() {
    OptimizationRemarkAnalysis R(kRemarkPass, "Site", siteLoc,
                                 &F.getEntryBlock());
    R << "Poseidon site " << ore::NV("Site", siteName) << ": ";
    if (tau > 0.0)
      R << "target " << ore::NV("Tau", num(tau));
    else
      R << "cost budget "
        << ore::NV("Budget",
                   costStr(scale.fromShare(Sess.Opts.Solver.CostBudget)));
    double asWritten = relErr(0.0);
    if (std::isfinite(asWritten))
      R << "; modelled error as written "
        << ore::NV("ErrorBefore", num(asWritten));
    if (nApplied == 0) {
      R << "; left as written";
    } else {
      double acc =
          selectedAcc().value_or(std::numeric_limits<double>::quiet_NaN());
      if (std::isfinite(relErr(acc)))
        R << ", after " << ore::NV("Error", num(relErr(acc)));
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

  for (const Unit &U : units) {
    if (!U.root)
      continue;
    ORE.emit([&]() {
      OptimizationRemarkAnalysis R(kRemarkPass, "Profile", U.root);
      if (U.kind == "matmul") {
        R << "Poseidon profile of the " << ore::NV("Shape", U.shape)
          << " matrix product: " << ore::NV("Executions", U.executions)
          << " multiply-adds";
        return R;
      }
      R << "Poseidon profile of " << ore::NV("Expr", exprWords(U));
      if (U.outC.size() > 1)
        R << " and " << ore::NV("MoreOutputs", (unsigned)U.outC.size() - 1)
          << " more output" << (U.outC.size() > 2 ? "s" : "");
      R << ": ";
      if (!U.inputRanges.empty())
        R << ore::NV("Ranges", joinWords(U.inputRanges)) << "; ";
      R << ore::NV("Executions", U.executions) << " executions";
      if (std::isfinite(U.sensitivity))
        R << ", sensitivity " << ore::NV("Sensitivity", num(U.sensitivity));
      return R;
    });
  }

  for (const Unit &U : units)
    for (const Alt &A : U.alts) {
      if (A.status != "applied" || !A.at)
        continue;
      std::string costChange;
      if (A.baseCost && A.costDelta)
        costChange = costStr(*A.baseCost) + " -> " +
                     costStr(*A.baseCost + *A.costDelta) +
                     pctStr(*A.baseCost, *A.baseCost + *A.costDelta);
      std::string errChange;
      if (A.accDelta && std::isfinite(share(A.baseAcc)))
        errChange = num(share(A.baseAcc)) + " -> " +
                    num(share(A.baseAcc + *A.accDelta));
      if (A.kind == "algebraic") {
        for (size_t k = 0; k < A.outputs.size(); ++k) {
          unsigned o = A.outputs[k];
          const Instruction *at = A.at;
          if (U.sg && o < U.sg->outputs.size())
            at = U.sg->outputs[o];
          std::string orig = o < U.outC.size() ? U.outC[o] : "";
          std::string now = k < A.c.size() ? A.c[k] : A.fpcore;
          ORE.emit([&]() {
            OptimizationRemark R(kRemarkPass, "Rewritten", at);
            R << "Poseidon rewrote " << ore::NV("Expr", clip(orig)) << " as "
              << ore::NV("Rewritten", clip(now));
            std::string how = A.approxNote;
            if (!A.approxRanges.empty())
              how +=
                  ", valid on the profiled range " + joinWords(A.approxRanges);
            if (!how.empty())
              R << " (" << ore::NV("Technique", how) << ")";
            if (!errChange.empty())
              R << "; modelled error " << ore::NV("Error", errChange);
            if (!costChange.empty())
              R << ", cost " << ore::NV("Cost", costChange);
            R << " [" << ore::NV("Id", A.id) << "]";
            return R;
          });
        }
      } else if (A.kind == "precision" || A.kind == "expansion") {
        ORE.emit([&]() {
          OptimizationRemark R(kRemarkPass, "PrecisionChanged", A.at);
          R << "Poseidon runs " << ore::NV("Ops", joinWords(A.changedOps))
            << " in " << ore::NV("Type", A.changedTo);
          if (A.c.size() == 1 && !A.c[0].empty())
            R << ": " << ore::NV("Rewritten", clip(A.c[0]));
          if (!errChange.empty())
            R << "; modelled error " << ore::NV("Error", errChange);
          if (!costChange.empty())
            R << ", cost " << ore::NV("Cost", costChange);
          R << " [" << ore::NV("Id", A.id) << "]";
          return R;
        });
      } else if (A.kind == "matmul") {
        ORE.emit([&]() {
          OptimizationRemark R(kRemarkPass, "MatmulRaised", A.at);
          R << "Poseidon raised the " << ore::NV("Shape", U.shape)
            << " matrix product to " << ore::NV("Technique", A.summary);
          if (A.domainError)
            R << "; modelled domain error "
              << ore::NV("Error", num(*A.domainError));
          if (!costChange.empty())
            R << ", cost " << ore::NV("Cost", costChange);
          R << " [" << ore::NV("Id", A.id) << "]";
          return R;
        });
      }
    }

  for (const Unit &U : units) {
    if (!U.root)
      continue;
    if (!U.applied)
      ORE.emit([&]() {
        OptimizationRemarkMissed R(kRemarkPass, "LeftAsWritten", U.root);
        R << "Poseidon left "
          << ore::NV("Expr", U.kind == "matmul"
                                 ? "the " + U.shape + " matrix product"
                                 : exprWords(U))
          << " as written: " << ore::NV("Reason", U.left);
        return R;
      });
    SmallVector<const Alt *, 8> rejected;
    for (const Alt &A : U.alts)
      if (A.status != "applied")
        rejected.push_back(&A);
    std::stable_sort(rejected.begin(), rejected.end(),
                     [](const Alt *a, const Alt *b) {
                       if (a->rank != b->rank)
                         return a->rank < b->rank;
                       if (a->err && b->err)
                         return *a->err < *b->err;
                       return false;
                     });
    for (size_t k = 0; k < rejected.size() && k < kMaxRejectedRemarks; ++k) {
      const Alt &A = *rejected[k];
      const Instruction *at = A.at ? A.at : U.root;
      ORE.emit([&]() {
        OptimizationRemarkMissed R(kRemarkPass, "Rejected", at);
        R << "Poseidon "
          << (A.status == "pareto" ? "did not select " : "rejected ")
          << ore::NV("Kind", kindWords(A.kind));
        if (!A.id.empty())
          R << " " << ore::NV("Id", A.id);
        if (A.c.size() == 1 && !A.c[0].empty())
          R << " to " << ore::NV("Rewritten", clip(A.c[0]));
        else if (A.kind != "algebraic")
          R << " (" << ore::NV("Rewritten", A.summary) << ")";
        else if (!A.fpcore.empty())
          R << " to " << ore::NV("Rewritten", clip(A.fpcore));
        R << ": " << ore::NV("Reason", A.reason);
        SmallVector<std::string, 2> facts;
        if (A.err && A.reason.find("modelled error") == std::string::npos)
          facts.push_back("modelled error " + num(*A.err));
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
  if (explanationDir(S.Opts).empty())
    return;
  json::Object result{{"status", status}, {"reason", reason}};
  if (!profile.empty())
    result["profile"] = profile;
  writeExplanation(
      S, F, body ? bodyProfileStem(S.Sites, F) : siteFileStem(S.Sites, F),
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

void remarkNoHerbie(Session &S, Function &F) {
  if (S.Herbie.LackReported)
    return;
  S.Herbie.LackReported = true;
  const std::string program = herbieProgram(S.Opts.Herbie);
  OptimizationRemarkEmitter ORE(&F);
  ORE.emit([&]() {
    OptimizationRemarkMissed R(kRemarkPass, "NoHerbie", siteLocation(S, F),
                               &F.getEntryBlock());
    R << "Poseidon proposes no algebraic rewrites: "
      << (program.empty() ? kBuiltWithoutHerbie.str()
                          : "there is no Herbie at " + program);
    return R;
  });
}

Error explainSolve(Session &S, Function &F, SiteState &st,
                   ArrayRef<SolutionStep> steps, const SolveRecord &rec) {
  const bool remarks =
      OptimizationRemarkEmitter::allowExtraAnalysis(F, kRemarkPass);
  const bool write = !explanationDir(S.Opts).empty();
  if (!remarks && !write)
    return Error::success();
  Explainer E(S, F, st, steps, rec);
  if (Error Err = E.build())
    return Err;
  if (remarks)
    E.emitRemarks();
  if (write)
    writeExplanation(S, F, siteFileStem(S.Sites, F), E.toJSON());
  return Error::success();
}

} // namespace poseidon
