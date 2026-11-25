//===- Herbie.cpp - algebraic rewrites from Herbie ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The algebraic units of an FP subgraph: Herbie's rewrites of its outputs,
// one output or an array of them per unit. In order: running Herbie, the
// cores of a subgraph, the accuracy of a rewrite (regime arms priced on their
// own samples), the cost of a rewrite, and the unit.
//
//===----------------------------------------------------------------------===//

#include "Herbie.h"
#include "Core/HerbieRun.h"
#include "Core/Parallel.h"
#include "IR/Utils.h"
#include "Model/CostModel.h"
#include "Model/Evaluators.h"
#include "Model/FPCore.h"
#include "Model/Sampling.h"
#include "Pass/Session.h"
#include "Targets/Target.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm;

namespace poseidon {

//===----------------------------------------------------------------------===//
// Running Herbie
//===----------------------------------------------------------------------===//

// The Herbie platform this compile searches against. It is the device's, so it
// is derived from the resolved cost model.
static Expected<const HerbiePlatform *> herbiePlatform(Session &S,
                                                       const Function &F) {
  HerbiePlatform &P = S.Herbie.Platform;
  if (S.Herbie.PlatformResolved)
    return &P;

  const std::string &NativeArch = getCostModelNativeArch(S.CostModel);
  const Target &T = Target::measuredOn(NativeArch);
  P.Name = T.herbiePlatform(NativeArch);

  if (!S.Opts.Herbie.HerbiePlatform.empty()) {
    // A value that looks like a file is the platform itself; anything else is
    // a platform name compiled into the Herbie binary. The name keeps naming
    // the device family either way, and the digest below distinguishes files.
    StringRef V(S.Opts.Herbie.HerbiePlatform);
    if (V.contains('/') || V.ends_with(".rkt"))
      P.Path = V.str();
    else
      P.Name = V.str();
  } else if (T.isDevice()) {
    // Generated next to the CSV it was priced from, by
    // `poseidon-calibrate --only herbie-platform`.
    StringRef Csv(costModelPath(S.CostModel));
    P.Path =
        (Csv.ends_with(".csv") ? Csv.drop_back(4) : Csv).str() + ".herbie.rkt";
    if (!sys::fs::exists(P.Path))
      return userError(
          F, F.getSubprogram(),
          Twine("the Herbie algebraic search needs the platform generated "
                "from this device's cost model, and ") +
              P.Path +
              " does not exist. Generate it with 'poseidon-calibrate --only "
              "herbie-platform --out " +
              costModelPath(S.CostModel) +
              "'. Another architecture's platform is that architecture's "
              "prices and is never substituted.");
  }

  if (!P.Path.empty()) {
    P.Digest = readFileDigest(P.Path);
    if (P.Digest.empty())
      return userError(F, F.getSubprogram(),
                       Twine("cannot read the Herbie platform ") + P.Path +
                           "; -poseidon-herbie-platform names a platform file "
                           "or a platform compiled into Herbie.");
  }
  S.Herbie.PlatformResolved = true;
  return &P;
}

// Herbie's --timeout has no unbounded value; this one stands for no limit of
// Poseidon's.
static constexpr unsigned kUnboundedCoreTimeout = 9999;

// Herbie's per-core --timeout: -poseidon-herbie-timeout, else the subgraph's
// wall-clock budget, which bounds it anyway.
static unsigned herbieCoreTimeout(const Options &O) {
  if (O.Herbie.HerbieTimeout)
    return O.Herbie.HerbieTimeout;
  if (O.Herbie.HerbieSubgraphTimeout)
    return O.Herbie.HerbieSubgraphTimeout;
  return kUnboundedCoreTimeout;
}

// How this compile runs Herbie: its -poseidon-herbie-* flags.
static Expected<HerbieRunOptions> herbieRunOptions(Session &S,
                                                   const Function &F) {
  const Options &O = S.Opts;
  HerbieRunOptions Opts;
  Opts.Program = O.Herbie.HerbieBinary.empty()
                     ? std::string(POSEIDON_HERBIE_DEFAULT_BINARY)
                     : O.Herbie.HerbieBinary;
  Opts.CacheOnly = !sys::fs::exists(Opts.Program);
  Expected<const HerbiePlatform *> Platform = herbiePlatform(S, F);
  if (!Platform)
    return Platform.takeError();
  Opts.Platform = **Platform;
  Opts.Seed = O.Model.RandomSeed;
  Opts.NumPoints = O.Herbie.HerbieNumPts;
  Opts.NumIters = O.Herbie.HerbieNumIters;
  Opts.NumEnodes = O.Herbie.HerbieNumEnodes;
  Opts.CoreTimeout = herbieCoreTimeout(O);
  Opts.NumThreads = O.Herbie.HerbieNumThreads;
  Opts.Processes = O.Herbie.HerbieProcesses;
  Opts.SubgraphTimeout = O.Herbie.HerbieSubgraphTimeout;
  Opts.CacheDir = O.Herbie.Cache;
  Opts.Print = O.Diagnostics.Print;
  return Opts;
}

namespace {
// A point of a core's cost-accuracy curve: [cost, error in bits].
struct HerbiePoint {
  double Cost = 0, ErrorBits = 0;
};

// An alternative Herbie lists for a core: [cost, error in bits, expression].
// An entry without the expression is no candidate.
struct HerbieAlternative {
  HerbiePoint Point;
  std::optional<std::string> Expr;
};

// A core's cost-accuracy list: [initial, best, alternatives].
struct HerbieCostAccuracy {
  HerbiePoint Initial, Best;
  std::vector<HerbieAlternative> Alternatives;
};

// What Herbie found for a core it has an expression for.
struct HerbieRewrites {
  std::string BestExpr;
  // The bits of the core's precision, which the errors are out of.
  double Bits = 0;
  HerbieCostAccuracy CostAccuracy;
};

// One test of a Herbie results file: the answer for the core it names.
struct HerbieCoreAnswer {
  std::string Name;
  // The core's index, which is its name.
  size_t Core = 0;
  // What Herbie reports about its search, printed when present.
  std::optional<std::string> Status;
  std::optional<double> Time, Start, End;
  // None when Herbie's output is "#f".
  std::optional<HerbieRewrites> Rewrites;
};

bool fromJSON(const json::Value &V, HerbiePoint &Point, json::Path P) {
  const json::Array *A = V.getAsArray();
  if (!A || A->size() < 2) {
    P.report("expected [cost, error]");
    return false;
  }
  return fromJSON((*A)[0], Point.Cost, P.index(0)) &&
         fromJSON((*A)[1], Point.ErrorBits, P.index(1));
}

bool fromJSON(const json::Value &V, HerbieAlternative &Alt, json::Path P) {
  if (!fromJSON(V, Alt.Point, P))
    return false;
  const json::Array &A = *V.getAsArray();
  if (A.size() < 3)
    return true;
  return fromJSON(A[2], Alt.Expr.emplace(), P.index(2));
}

bool fromJSON(const json::Value &V, HerbieCostAccuracy &CA, json::Path P) {
  const json::Array *A = V.getAsArray();
  if (!A || A->size() < 3) {
    P.report("expected [initial, best, alternatives]");
    return false;
  }
  return fromJSON((*A)[0], CA.Initial, P.index(0)) &&
         fromJSON((*A)[1], CA.Best, P.index(1)) &&
         fromJSON((*A)[2], CA.Alternatives, P.index(2));
}

bool fromJSON(const json::Value &V, HerbieCoreAnswer &Answer, json::Path P) {
  json::ObjectMapper O(V, P);
  std::string Output;
  if (!O || !O.map("name", Answer.Name) || !O.map("output", Output))
    return false;
  const json::Object &Test = *V.getAsObject();
  if (std::optional<StringRef> Status = Test.getString("status"))
    Answer.Status = Status->str();
  Answer.Time = Test.getNumber("time");
  Answer.Start = Test.getNumber("start");
  Answer.End = Test.getNumber("end");
  if (Output == "#f")
    return true;
  HerbieRewrites &R = Answer.Rewrites.emplace();
  R.BestExpr = std::move(Output);
  return O.map("bits", R.Bits) && O.map("cost-accuracy", R.CostAccuracy);
}
} // namespace

// Reads the Herbie results file `Contents`, the answers for `NumCores` cores
// named by their index.
static Expected<std::vector<HerbieCoreAnswer>>
readHerbieResults(StringRef Contents, size_t NumCores) {
  Expected<json::Value> Results = json::parse(Contents);
  if (!Results)
    return Results.takeError();
  const json::Object *Top = Results->getAsObject();
  const json::Array *Tests = Top ? Top->getArray("tests") : nullptr;
  if (!Tests)
    return createStringError(
        inconvertibleErrorCode(),
        "expected an object with a \"tests\" array at the top level");
  std::vector<HerbieCoreAnswer> Answers(Tests->size());
  for (auto [I, Test] : enumerate(*Tests)) {
    HerbieCoreAnswer &Answer = Answers[I];
    std::string Where = formatv("tests[{0}]", I);
    json::Path::Root Root(Where);
    if (!fromJSON(Test, Answer, Root)) {
      std::string Why = toString(Root.getError());
      if (!Answer.Name.empty())
        Why += ", core \"" + Answer.Name + "\"";
      return createStringError(inconvertibleErrorCode(), Why);
    }
    if (StringRef(Answer.Name).getAsInteger(10, Answer.Core) ||
        Answer.Core >= NumCores)
      return createStringError(
          inconvertibleErrorCode(),
          formatv("{0} names core \"{1}\", and the cores asked about are 0 "
                  "to {2}",
                  Where, Answer.Name, NumCores - 1));
  }
  return Answers;
}

namespace {

// Herbie's answers for one invocation's cores: each run at a Taylor-term
// count, and the answers of those that answered with where each came from,
// its run's label and cache file.
struct HerbieAnswers {
  SmallVector<HerbieRunResult, 2> runs;
  SmallVector<std::vector<HerbieCoreAnswer>, 2> answers;
  SmallVector<std::pair<std::string, std::string>, 2> sources;
};

} // namespace

// Runs Herbie on `cores`, once per -poseidon-herbie-num-taylor-terms count,
// and reads what each run answered. A Herbie that cannot run is an error at
// the site `F`; under -poseidon-strict-mode so are a crash and an unreadable
// answer, which otherwise mean no candidate from that run.
static Expected<HerbieAnswers> askHerbie(Session &S,
                                         const std::vector<std::string> &cores,
                                         StringRef label, const Function &F) {
  const Options &O = S.Opts;
  if (!O.Herbie.HerbieBinary.empty() && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] using Herbie binary '" << O.Herbie.HerbieBinary
                 << "'\n";
  Expected<HerbieRunOptions> RunOpts = herbieRunOptions(S, F);
  if (!RunOpts)
    return RunOpts.takeError();
  HerbieRunOptions &Opts = *RunOpts;

  SmallVector<std::optional<unsigned>, 2> TermCounts(
      O.Herbie.HerbieNumTaylorTerms.begin(),
      O.Herbie.HerbieNumTaylorTerms.end());
  if (TermCounts.empty())
    TermCounts.push_back(std::nullopt);
  HerbieAnswers H;
  for (std::optional<unsigned> Terms : TermCounts) {
    Opts.TaylorTerms = Terms;
    const std::string RunLabel =
        Terms ? (label + " at " + Twine(*Terms) + " Taylor terms").str()
              : label.str();
    Expected<HerbieRunResult> Run =
        runHerbie(cores, Opts, RunLabel, S.Herbie.HerbieDigests);
    if (!Run)
      return userError(F, F.getSubprogram(), toString(Run.takeError()));
    const HerbieRunResult &R = H.runs.emplace_back(std::move(*Run));
    // A timeout is Herbie's answer for these cores; a Herbie that does not
    // start gives none.
    if (R.Result == HerbieRunResult::Outcome::Failed)
      return userError(
          F, F.getSubprogram(),
          Twine(Opts.CacheDir.empty() ? "no cache is kept"
                                      : "the cache " + Opts.CacheDir +
                                            " holds no Herbie answer") +
              " for " + RunLabel +
              (Opts.Program.empty()
                   ? ", and this Poseidon was built without Herbie"
                   : ", and the Herbie at " + Opts.Program + " cannot run (" +
                         R.Failure + ")") +
              ". Point -poseidon-herbie-binary at a Herbie, or turn algebraic "
              "rewriting off with -poseidon-enable-herbie=0.");
    if (!R.Warning.empty())
      diagnose(F, F.getSubprogram(), R.Warning, DS_Warning);
    if (R.Result == HerbieRunResult::Outcome::Crashed && O.Model.StrictMode)
      return userError(F, F.getSubprogram(),
                       Twine("strict mode: Herbie crashed (") + R.Crash +
                           ") on " + RunLabel +
                           ". Without -poseidon-strict-mode the subgraph is "
                           "left as written.");
    if (R.Result != HerbieRunResult::Outcome::Ok)
      continue;
    const bool cached = R.From == HerbieRunResult::Origin::Cache;
    StringRef Origin = cached ? "the cached answer" : "Herbie's answer";
    // An answer that cannot be read is no answer, as a crash of Herbie is.
    Expected<std::vector<HerbieCoreAnswer>> Read =
        readHerbieResults(R.ResultsJson, cores.size());
    if (!Read) {
      std::string Why = toString(Read.takeError());
      if (O.Model.StrictMode)
        return userError(F, F.getSubprogram(),
                         "strict mode: " + Origin + " for " + RunLabel +
                             " is not a Herbie results file (" + Why +
                             "); without -poseidon-strict-mode the subgraph "
                             "is left as written");
      diagnose(F, F.getSubprogram(),
               Origin + " for " + RunLabel + " is not a Herbie results file (" +
                   Why +
                   "), so the subgraph has no algebraic candidate from it",
               DS_Warning);
      continue;
    }
    H.answers.push_back(std::move(*Read));
    H.sources.push_back({RunLabel, R.CacheFile});
    ++(cached ? S.Herbie.Cached : S.Herbie.Fresh);
  }
  return H;
}

// Records `answer` in its core's unit `RU`: the status of the first answer,
// and each expression not in `seen` as a candidate, priced relative to the
// core as written.
static void addAnswer(const HerbieCoreAnswer &answer, StringRef status,
                      RewriteUnit &RU, std::unordered_set<std::string> &seen) {
  if (!RU.herbieAnswered)
    RU.herbieStatus = status.str();
  if (!answer.Rewrites)
    return;
  RU.herbieAnswered = true;

  const HerbieRewrites &rewrites = *answer.Rewrites;
  const HerbieCostAccuracy &costAccuracy = rewrites.CostAccuracy;
  const double bits = rewrites.Bits;
  RU.herbieBits = bits;
  const double initialCost = costAccuracy.Initial.Cost;
  RU.initialHerbieAccuracy = 1.0 - costAccuracy.Initial.ErrorBits / bits;

  auto add = [&](const HerbiePoint &point, const std::string &expr) {
    if (!seen.insert(expr).second)
      return;
    RewriteCandidate candidate(point.Cost / initialCost,
                               1.0 - point.ErrorBits / bits, expr);
    candidate.ordinal = RU.candidates.size();
    RU.candidates.push_back(candidate);
  };
  add(costAccuracy.Best, rewrites.BestExpr);
  for (const HerbieAlternative &alternative : costAccuracy.Alternatives)
    if (alternative.Expr)
      add(alternative.Point, *alternative.Expr);
}

// An error at the site `F` if an expression of `answer`, for the unit `RU`
// and read from `source` (its run's label and cache file), is one the model
// cannot read: an answer is external input, checked before anything evaluates
// or lowers it.
static Error checkAnswer(const HerbieCoreAnswer &answer,
                         const std::pair<std::string, std::string> &source,
                         const RewriteUnit &RU, const Function &F, FPGraph &G) {
  auto check = [&](const std::string &expr) -> Error {
    Expected<FPNode *> parsed =
        parseHerbieExpr(expr, G, nullptr, &RU.subgraph->inputs);
    if (parsed)
      return Error::success();
    return userError(
        F, F.getSubprogram(),
        "the Herbie answer for " + source.first + " in " +
            (source.second.empty() ? std::string("no cache file")
                                   : source.second) +
            " has, for core \"" + answer.Name + "\", " +
            toString(parsed.takeError()) +
            "; Poseidon cannot model that expression. Remove the "
            "file to search again, or turn algebraic rewriting off "
            "with -poseidon-enable-herbie=0.");
  };
  if (Error E = check(answer.Rewrites->BestExpr))
    return E;
  for (const HerbieAlternative &alternative :
       answer.Rewrites->CostAccuracy.Alternatives)
    if (alternative.Expr)
      if (Error E = check(*alternative.Expr))
        return E;
  return Error::success();
}

// Collects Herbie's rewrites of `cores`, cores of the subgraph `subgraphIdx`
// of the site `F` over the graph `G`, into their units `COs`, unpriced.
static Error collectHerbieRewrites(Session &S,
                                   const std::vector<std::string> &cores,
                                   std::vector<RewriteUnit> &COs,
                                   int subgraphIdx, const Function &F,
                                   FPGraph &G) {
  const std::string label =
      "subgraph " + std::to_string(subgraphIdx) + " of " + F.getName().str();
  Expected<HerbieAnswers> Asked = askHerbie(S, cores, label, F);
  if (!Asked)
    return Asked.takeError();
  HerbieAnswers &H = *Asked;

  const bool print = S.Opts.Diagnostics.Print;
  std::vector<std::unordered_set<std::string>> seenExprs(COs.size());
  for (auto [answers, source] : zip(H.answers, H.sources))
    for (const HerbieCoreAnswer &answer : answers) {
      StringRef status = answer.Status ? StringRef(*answer.Status) : "?";
      if (print) {
        llvm::errs() << "[poseidon] Herbie core \"" << answer.Name
                     << "\" (subgraph " << subgraphIdx << " of " << F.getName()
                     << "): status=" << status;
        if (answer.Time)
          llvm::errs() << " time=" << (*answer.Time / 1000.0) << 's';
        if (answer.Start)
          llvm::errs() << " start-err-bits=" << *answer.Start;
        if (answer.End)
          llvm::errs() << " end-err-bits=" << *answer.End;
        llvm::errs() << '\n';
        if (!answer.Rewrites)
          llvm::errs() << "[poseidon] Herbie core \"" << answer.Name
                       << "\": no output expression (status " << status
                       << "); no algebraic candidate from this core\n";
      }
      if (answer.Rewrites)
        if (Error E = checkAnswer(answer, source, COs[answer.Core], F, G))
          return E;
      addAnswer(answer, status, COs[answer.Core], seenExprs[answer.Core]);
    }

  for (const HerbieRunResult &Run : H.runs) {
    if (Run.Result != HerbieRunResult::Outcome::Timeout &&
        Run.Result != HerbieRunResult::Outcome::Crashed)
      continue;
    for (RewriteUnit &RU : COs)
      if (!RU.herbieAnswered)
        RU.herbieStatus = Run.Result == HerbieRunResult::Outcome::Timeout
                              ? "timeout"
                              : "crashed";
  }
  return Error::success();
}

void reportHerbieAnswers(Session &S) {
  HerbieRunState &H = S.Herbie;
  if (!H.Cached && !H.Fresh)
    return;
  if (S.Opts.Diagnostics.Print)
    llvm::errs() << "[poseidon] Herbie answers: " << H.Fresh << " fresh, "
                 << H.Cached << " from the cache\n";
  H.Cached = H.Fresh = 0;
}

//===----------------------------------------------------------------------===//
// The cores of a subgraph
//===----------------------------------------------------------------------===//

// The precisions every output is searched at, in the order their cores are
// sent.
static constexpr const char *kHerbiePrecisions[] = {"binary64", "binary32"};

// The FPCore Herbie searches: `body` over `args` at `prec`, under `pre`, with
// the properties `extra` and the name `name`.
static std::string fpcoreText(const SmallSet<std::string, 8> &args,
                              StringRef prec, StringRef pre, StringRef extra,
                              size_t name, StringRef body) {
  return (Twine("(FPCore (") + join(args, " ") +
          ") :herbie-conversions ([binary64 binary32]) :precision " + prec +
          " :pre " + pre + extra + " :name \"" + Twine(name) + "\" " + body +
          ")")
      .str();
}

// v with every digit a double needs, so that it round-trips.
static std::string roundTripDigits(double v) {
  std::string s;
  raw_string_ostream(s) << format("%.17e", v);
  return s;
}

// A profiled range narrower than this is widened around its midpoint, by at
// least kPointHalfWidth and by kPointRelativeHalfWidth of the midpoint: Herbie
// samples no point of an empty or one-point range.
static constexpr double kPointWidth = 1e-10;
static constexpr double kPointHalfWidth = 1e-10;
static constexpr double kPointRelativeHalfWidth = 1e-6;

// The precondition of a core over `args`: each within its profiled range,
// and, under -poseidon-herbie-pre-floor-bits, away from zero.
static std::string rangePrecondition(const Options &O,
                                     const SmallSet<std::string, 8> &args,
                                     const FPGraph &G) {
  std::string preconditions;
  for (const std::string &arg : args) {
    const FPNode &node = *G.valueToNodeMap.at(G.symbolToValueMap.at(arg));
    double lower = node.getLowerBound();
    double upper = node.getUpperBound();

    if (upper - lower < kPointWidth && !std::isinf(lower) &&
        !std::isinf(upper)) {
      double midpoint = (lower + upper) / 2.0;
      double tolerance = std::max(kPointHalfWidth,
                                  std::abs(midpoint) * kPointRelativeHalfWidth);
      lower = midpoint - tolerance;
      upper = midpoint + tolerance;
    }
    const std::string lowerStr = roundTripDigits(lower),
                      upperStr = roundTripDigits(upper);

    if (O.Herbie.HerbiePreFloorBits > 0 && lower < 0 && upper > 0 &&
        std::isfinite(lower) && std::isfinite(upper)) {
      const std::string floorStr = roundTripDigits(
          std::ldexp(std::max(-lower, upper),
                     -static_cast<int>(O.Herbie.HerbiePreFloorBits)));
      preconditions += " (or (<= " + lowerStr + " " + arg + " -" + floorStr +
                       ") (<= " + floorStr + " " + arg + " " + upperStr + "))";
      continue;
    }

    preconditions +=
        " (<=" +
        (std::isinf(lower) ? (lower > 0 ? " INFINITY" : " (- INFINITY)")
                           : (" " + lowerStr)) +
        " " + arg +
        (std::isinf(upper) ? (upper > 0 ? " INFINITY" : " (- INFINITY)")
                           : (" " + upperStr)) +
        ")";
  }
  return preconditions.empty() ? "TRUE" : "(and" + preconditions + ")";
}

// The outputs' expressions with every subterm used more than once bound by
// let*, so Herbie reads the subgraph as the DAG it is rather than one tree per
// output.
static Expected<std::string> arrayCoreBody(ArrayRef<Instruction *> outputs,
                                           FPGraph &G,
                                           const SetVector<Value *> &inputs) {
  auto isLeaf = [&](FPNode *N) {
    if (isa<FPConst>(N))
      return true;
    auto *LL = cast<FPLLValue>(N);
    return inputs.contains(LL->value);
  };
  DenseMap<FPNode *, unsigned> uses;
  SmallVector<FPNode *, 32> order;
  SmallPtrSet<FPNode *, 32> visited;
  // The leaves are spelled out first: spelling one out can fail.
  DenseMap<FPNode *, std::string> leafText;
  Error leafError = Error::success();
  auto spell = [&](FPNode *N) {
    if (leafError || leafText.count(N))
      return;
    Expected<std::string> text = N->fullExpression(G, inputs);
    if (!text) {
      leafError = text.takeError();
      return;
    }
    leafText[N] = std::move(*text);
  };
  std::function<void(FPNode *)> walk = [&](FPNode *N) {
    if (isLeaf(N))
      spell(N);
    if (!visited.insert(N).second || isLeaf(N))
      return;
    for (auto &operand : N->operands) {
      if (!isLeaf(operand))
        ++uses[operand];
      walk(operand);
    }
    order.push_back(N);
  };
  for (Instruction *output : outputs) {
    FPNode *N = G.valueToNodeMap[output];
    ++uses[N];
    walk(N);
  }
  if (leafError)
    return std::move(leafError);

  DenseMap<FPNode *, std::string> names;
  std::function<std::string(FPNode *)> ref;
  auto define = [&](FPNode *N) {
    std::string expr =
        "(" + (N->is(fpcore::Neg) ? std::string("-") : operationName(*N));
    for (auto &operand : N->operands)
      expr += " " + ref(operand);
    return expr + ")";
  };
  ref = [&](FPNode *N) -> std::string {
    if (isLeaf(N))
      return leafText.lookup(N);
    auto It = names.find(N);
    if (It != names.end())
      return It->second;
    return define(N);
  };

  std::string bindings;
  for (FPNode *N : order) {
    if (uses[N] < 2)
      continue;
    std::string def = define(N);
    std::string name = "t" + std::to_string(names.size() + 1);
    if (!bindings.empty())
      bindings += " ";
    bindings += "(" + name + " " + def + ")";
    names[N] = name;
  }
  std::string body = "(array";
  for (Instruction *output : outputs)
    body += " " + ref(G.valueToNodeMap[output]);
  body += ")";
  return bindings.empty() ? body : "(let* (" + bindings + ") " + body + ")";
}

namespace {

// An output of a subgraph that Herbie is asked to rewrite.
struct EligibleOutput {
  Instruction *output;
  std::string expr;
  double grad;
  uint64_t executions;
};

// The cores of one subgraph, as the FPCore text sent to Herbie and the unit
// each becomes.
struct CoreBatch {
  std::vector<std::string> inputs;
  std::vector<RewriteUnit> units;
};

// The cores Herbie is asked for one subgraph: one per output and precision,
// or, under -poseidon-herbie-arrays, one per group of outputs and precision.
class SubgraphCores {
public:
  SubgraphCores(Session &S, Subgraph &subgraph, int subgraphIdx, FPGraph &G)
      : S(S), O(S.Opts), subgraph(subgraph), subgraphIdx(subgraphIdx), G(G) {}

  // Names each input of the subgraph that is no constant by a symbol, the
  // next of `symbolCounter` for one without.
  void nameInputs(int &symbolCounter);

  // `output` as an output Herbie is asked about, unless its gradient is zero
  // or its expression is a single operation or too long.
  Expected<std::optional<EligibleOutput>> eligible(Instruction *output) const;

  // A core of `E` at each of `precs`, into `batch`.
  Error addPerOutputCores(const EligibleOutput &E, ArrayRef<const char *> precs,
                          CoreBatch &batch) const;

  // The groups of `eligible` that form array cores: outputs that read the
  // same inputs (hence the same precondition) and execute equally often, of
  // two outputs or more.
  SmallVector<SmallVector<const EligibleOutput *, 8>, 2>
  arrayGroups(ArrayRef<EligibleOutput> eligible) const;

  // A core of `group` at each precision, into `batch`.
  Error addArrayCores(ArrayRef<const EligibleOutput *> group,
                      CoreBatch &batch) const;

private:
  // The :herbie-weights property of an array core of `outputs`, which weighs
  // each output by its profiled sensitivity; empty for none.
  std::string arrayWeights(ArrayRef<Instruction *> outputs) const;

  Session &S;
  const Options &O;
  Subgraph &subgraph;
  int subgraphIdx;
  FPGraph &G;
};

} // namespace

void SubgraphCores::nameInputs(int &symbolCounter) {
  for (Value *input : subgraph.inputs) {
    FPNode &node = *G.valueToNodeMap[input];
    if (isa<FPConst>(node))
      continue;
    if (!node.hasSymbol())
      node.symbol = "v" + std::to_string(symbolCounter++);
    G.symbolToValueMap[node.symbol] = input;
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] assigning symbol: " << node.symbol << " to "
                   << *input << '\n';
  }
}

Expected<std::optional<EligibleOutput>>
SubgraphCores::eligible(Instruction *output) const {
  FPNode &node = *G.valueToNodeMap[output];
  if (node.grad == 0.) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] Skipping zero gradient instruction: "
                   << *output << '\n';
    return std::nullopt;
  }

  Expected<std::string> spelled = node.fullExpression(G, subgraph.inputs);
  if (!spelled)
    return spelled.takeError();
  std::string &expr = *spelled;
  if (expr.length() > O.Candidates.MaxExprLength) {
    diagnose(*output->getFunction(), output->getDebugLoc(),
             "an output expression of " + Twine(expr.length()) +
                 " characters is over -poseidon-max-expr-length=" +
                 Twine(O.Candidates.MaxExprLength) +
                 " and is not sent to Herbie",
             DS_Warning);
    return std::nullopt;
  }

  const size_t parenCount = count(expr, '(');
  assert(parenCount > 0 && "an output expression is an FPCore operation");
  if (parenCount == 1) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] Skipping Herbie for simple expression: "
                   << expr << '\n';
    return std::nullopt;
  }
  return EligibleOutput{output, std::move(expr), node.grad, node.executions};
}

Error SubgraphCores::addPerOutputCores(const EligibleOutput &E,
                                       ArrayRef<const char *> precs,
                                       CoreBatch &batch) const {
  SmallSet<std::string, 8> args;
  getUniqueArgs(E.expr, args);
  std::string precondition = rangePrecondition(O, args, G);
  for (const char *prec : precs) {
    RewriteUnit RU(subgraph, {E.output}, {E.grad}, E.expr, E.executions);
    if (Error Err = RU.priceAsWritten(S.CostModel))
      return Err;
    RU.herbiePrecision = prec;
    std::string herbieInput =
        fpcoreText(args, prec, precondition, "", batch.units.size(), E.expr);
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] Herbie input:\n" << herbieInput << '\n';
    batch.inputs.push_back(herbieInput);
    batch.units.push_back(RU);
  }
  return Error::success();
}

SmallVector<SmallVector<const EligibleOutput *, 8>, 2>
SubgraphCores::arrayGroups(ArrayRef<EligibleOutput> eligible) const {
  auto groupKey = [&](const EligibleOutput &E) {
    std::string key = std::to_string(E.executions);
    if (O.Herbie.HerbieArrayGrouping == "inputs") {
      SmallSet<std::string, 8> args;
      getUniqueArgs(E.expr, args);
      std::vector<std::string> sorted(args.begin(), args.end());
      sort(sorted);
      for (const std::string &arg : sorted)
        key += " " + arg;
    }
    return key;
  };
  MapVector<std::string, SmallVector<const EligibleOutput *, 8>,
            std::unordered_map<std::string, unsigned>>
      byKey;
  for (const EligibleOutput &E : eligible)
    byKey[groupKey(E)].push_back(&E);
  SmallVector<SmallVector<const EligibleOutput *, 8>, 2> groups;
  for (auto &[key, group] : byKey)
    if (group.size() >= 2)
      groups.push_back(std::move(group));
  return groups;
}

std::string SubgraphCores::arrayWeights(ArrayRef<Instruction *> outputs) const {
  if (O.Herbie.HerbieArrayWeights == "none")
    return "";
  SmallVector<double, 8> sens;
  for (Instruction *output : outputs) {
    const FPNode &N = *G.valueToNodeMap[output];
    if (O.Herbie.HerbieArrayWeights == "gradient") {
      double lo = N.getLowerBound(), hi = N.getUpperBound();
      double meanMag = lo >= 0 || hi <= 0
                           ? (std::fabs(lo) + std::fabs(hi)) / 2
                           : (lo * lo + hi * hi) / (2 * (hi - lo));
      sens.push_back(std::fabs(N.grad) * meanMag);
    } else {
      sens.push_back(N.sens);
    }
  }
  double total = 0.0;
  for (double v : sens)
    total += v;
  if (!(total > 0.0 && std::isfinite(total) &&
        all_of(sens, [](double v) { return v >= 0.0; }))) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] an array core of subgraph " << subgraphIdx
                   << " is sent unweighted: its outputs' profiled "
                      "sensitivities do not sum to a positive value\n";
    return "";
  }
  std::string weights = " :herbie-weights (";
  raw_string_ostream os(weights);
  ListSeparator sep(" ");
  for (double v : sens)
    os << sep << format("%.6e", v * sens.size() / total);
  os << ")";
  return weights;
}

Error SubgraphCores::addArrayCores(ArrayRef<const EligibleOutput *> group,
                                   CoreBatch &batch) const {
  SmallVector<Instruction *, 8> outputs;
  SmallVector<Value *, 8> roots;
  SmallVector<double, 8> grads;
  for (const EligibleOutput *E : group) {
    outputs.push_back(E->output);
    roots.push_back(E->output);
    grads.push_back(E->grad);
  }
  Expected<std::string> core = arrayCoreBody(outputs, G, subgraph.inputs);
  if (!core)
    return core.takeError();
  const std::string &body = *core;
  SmallSet<std::string, 8> args;
  getUniqueArgs(body, args);
  std::string precondition = rangePrecondition(O, args, G);
  std::string weights = arrayWeights(outputs);
  for (const char *prec : kHerbiePrecisions) {
    std::string herbieInput =
        fpcoreText(args, prec, precondition, weights, batch.units.size(), body);
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] Herbie array input:\n" << herbieInput << '\n';
    batch.inputs.push_back(herbieInput);
    RewriteUnit &RU =
        batch.units.emplace_back(subgraph, ArrayRef<Value *>(roots), grads,
                                 body, group.front()->executions);
    if (Error Err = RU.priceAsWritten(S.CostModel))
      return Err;
    RU.herbiePrecision = prec;
  }
  return Error::success();
}

Error proposeHerbieCandidates(Session &S, Function &F, Subgraph &subgraph,
                              int subgraphCounter, int &symbolCounter,
                              FPGraph &G, SmallVectorImpl<RewriteUnit> &COs) {
  SubgraphCores cores(S, subgraph, subgraphCounter, G);
  cores.nameInputs(symbolCounter);

  // The cores of one output each, searched in their own invocation.
  CoreBatch perOutput;
  SmallVector<EligibleOutput, 8> eligible;
  assert(!subgraph.outputs.empty() && "No outputs found for subgraph");
  for (Instruction *output : subgraph.outputs) {
    Expected<std::optional<EligibleOutput>> E = cores.eligible(output);
    if (!E)
      return E.takeError();
    if (!*E)
      continue;
    if (S.Opts.Herbie.HerbieArrays)
      eligible.push_back(std::move(**E));
    else if (Error Err =
                 cores.addPerOutputCores(**E, kHerbiePrecisions, perOutput))
      return Err;
  }

  // The array cores, then the outputs no array covers, are searched in one
  // invocation; per-output cores stand in for an array core that returned
  // nothing, in the other.
  CoreBatch arrays;
  // The group of each array core.
  SmallVector<const SmallVector<const EligibleOutput *, 8> *, 4> groupOf;
  SmallVector<SmallVector<const EligibleOutput *, 8>, 2> groups =
      cores.arrayGroups(eligible);
  for (const SmallVector<const EligibleOutput *, 8> &group : groups) {
    if (Error Err = cores.addArrayCores(group, arrays))
      return Err;
    groupOf.append(std::size(kHerbiePrecisions), &group);
  }
  const size_t numArrayCores = arrays.units.size();
  auto inArray = [&](const EligibleOutput &E) {
    return any_of(groups, [&](const SmallVector<const EligibleOutput *, 8> &G) {
      return is_contained(G, &E);
    });
  };
  for (const EligibleOutput &E : eligible)
    if (!inArray(E))
      if (Error Err = cores.addPerOutputCores(
              E, kHerbiePrecisions, numArrayCores ? arrays : perOutput))
        return Err;
  if (numArrayCores)
    if (Error Err = collectHerbieRewrites(S, arrays.inputs, arrays.units,
                                          subgraphCounter, F, G))
      return Err;
  for (const EligibleOutput &E : eligible) {
    SmallVector<const char *, 2> sendAt;
    for (size_t k = 0; k < numArrayCores; ++k)
      if (is_contained(*groupOf[k], &E) && !arrays.units[k].herbieAnswered)
        sendAt.push_back(kHerbiePrecisions[k % std::size(kHerbiePrecisions)]);
    if (sendAt.empty())
      continue;
    if (S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] Herbie subgraph " << subgraphCounter
                   << ": array core returned nothing for " << *E.output
                   << "; sending it separately\n";
    if (Error Err = cores.addPerOutputCores(E, sendAt, perOutput))
      return Err;
  }
  COs.append(std::make_move_iterator(arrays.units.begin()),
             std::make_move_iterator(arrays.units.end()));

  if (!perOutput.inputs.empty()) {
    if (Error Err = collectHerbieRewrites(S, perOutput.inputs, perOutput.units,
                                          subgraphCounter, F, G))
      return Err;
    COs.append(std::make_move_iterator(perOutput.units.begin()),
               std::make_move_iterator(perOutput.units.end()));
  }
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Accuracy of a rewrite
//===----------------------------------------------------------------------===//

// An `if` of a regime-split candidate and the arm a sample takes.
using RegimeArm = std::pair<const FPNode *, bool>;

static void collectIfNodes(const FPNode *node,
                           SmallPtrSetImpl<const FPNode *> &seen,
                           SmallVectorImpl<const FPNode *> &ifs) {
  if (!seen.insert(node).second)
    return;
  if (node->is(fpcore::If))
    ifs.push_back(node);
  for (const auto &operand : node->operands)
    collectIfNodes(operand, seen, ifs);
}

// The arms the candidate actually evaluates at `point`, following each `if` the
// way the materialized select resolves it. `ev` evaluates at `point`.
static void collectTakenArms(const FPNode *node,
                             const MapVector<Value *, double> &point,
                             FPEvaluator &ev,
                             SmallPtrSetImpl<const FPNode *> &seen,
                             SmallVectorImpl<RegimeArm> &arms) {
  if (!seen.insert(node).second)
    return;
  if (!node->is(fpcore::If)) {
    for (const auto &operand : node->operands)
      collectTakenArms(operand, point, ev, seen, arms);
    return;
  }
  const FPNode *cond = node->operands[0];
  ev.evaluateNode(cond, point);
  bool taken = ev.getResult(cond) == 1.0;
  arms.push_back({node, taken});
  collectTakenArms(cond, point, ev, seen, arms);
  collectTakenArms(node->operands[taken ? 1 : 2], point, ev, seen, arms);
}

static SmallVector<RegimeArm, 4>
takenArms(const FPNode *root, const MapVector<Value *, double> &point,
          FPEvaluator &ev) {
  SmallPtrSet<const FPNode *, 32> seen;
  SmallVector<RegimeArm, 4> arms;
  collectTakenArms(root, point, ev, seen, arms);
  return arms;
}

namespace {

struct ArmStats {
  double candSum = 0.0;
  double origSum = 0.0;
  unsigned count = 0;
};

// One element of a candidate, priced against its output's original on the
// subgraph's samples. The pricing of an element reads nothing another element
// writes, so the elements of a unit are priced concurrently.
struct ElementPricing {
  FPNode *node;
  // The output's original expression, its values at the samples, and its
  // mean error there.
  const FPNode *orig = nullptr;
  const SubgraphSamples::Reference *ref = nullptr;
  double origCost = 0.0;
  const std::string *exprForLog = nullptr;
  SmallVector<const FPNode *, 2> ifNodes;

  std::map<RegimeArm, ArmStats> armStats;
  bool discard = false;
  std::string why;
  double cost = 0.0;
  size_t drawn = 0;
  bool searching = false;
  // Its -poseidon-print lines, printed in candidate order.
  std::string diag;

  // Whether `arm` has fewer samples than -poseidon-min-arm-samples.
  bool starved(const Options &O, const RegimeArm &arm) {
    return armStats[arm].count < O.Model.MinArmSamples;
  }
  // Whether any arm has.
  bool starved(const Options &O) const {
    return any_of(armStats, [&](const auto &kv) {
      return kv.second.count < O.Model.MinArmSamples;
    });
  }
};

} // namespace

// The element on the samples every candidate is priced on.
static void priceOnPoints(const Options &O, ElementPricing &E,
                          ArrayRef<SubgraphSamples::Point> points) {
  const size_t n = points.size();
  SmallVector<double, 0> values(n);
  SmallVector<SmallVector<RegimeArm, 4>, 0> arms(E.ifNodes.empty() ? 0 : n);
  forEachBlock(O.Model.Threads, n, 64, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      FPEvaluator ev;
      ev.evaluateNode(E.node, points[i]);
      values[i] = ev.getResult(E.node);
      if (!E.ifNodes.empty())
        arms[i] = takenArms(E.node, points[i], ev);
    }
  });

  double sum = 0.0;
  unsigned count = 0;
  unsigned broken = 0;
  for (size_t i = 0; i < n; ++i) {
    double realVal = values[i];
    double goldVal = E.ref->gold[i];
    if (std::isfinite(goldVal) && goldVal != 0.0) {
      double origRel =
          std::fabs(E.ref->value[i] - goldVal) / std::fabs(goldVal);
      double candRel = std::fabs(realVal - goldVal) / std::fabs(goldVal);
      if (origRel < 1.0 && !(candRel < 1.0))
        ++broken;
    }

    if (std::isfinite(goldVal) && !std::isfinite(realVal)) {
      E.discard = true;
      E.why = "it returns a non-finite value at a sampled input where the "
              "exact result is finite";
      break;
    }

    double error = sampleError(goldVal, realVal);
    if (!std::isnan(error)) {
      sum += error;
      ++count;
      double origError = E.ref->error[i];
      if (!E.ifNodes.empty() && !std::isnan(origError))
        for (const RegimeArm &arm : arms[i]) {
          ArmStats &st = E.armStats[arm];
          st.candSum += error;
          st.origSum += origError;
          ++st.count;
        }
    }
  }
  if (!E.discard) {
    if (count == 0) {
      E.discard = true;
      E.why = "every sample of it scored NaN";
    } else {
      E.cost = sum / count;
    }
  }
  if (!E.discard && O.Model.MaxBrokenShare >= 0.0 &&
      broken > O.Model.MaxBrokenShare * n) {
    if (O.Diagnostics.Print)
      raw_string_ostream(E.diag)
          << "[poseidon] dropping candidate: relative error >= 1 on " << broken
          << " of " << n << " samples: " << E.exprForLog->substr(0, 120)
          << '\n';
    E.discard = true;
    E.why = formatv("its relative error is 1 or more on {0} of {1} samples",
                    broken, n)
                .str();
  }
  E.searching = !E.discard && !E.ifNodes.empty() && E.starved(O) &&
                O.Model.ArmSearchFactor > 0 && O.Model.NumSamples > 0;
}

// Uniform marginal sampling weights each regime by its share of the profiled
// box, which need not be its share of the workload: correlated inputs can put
// most of the run in an arm the box almost never reaches. Every arm is priced
// on its own samples, drawn round by round until it has enough or the budget
// is spent. An element takes the rounds in order and at each point reads only
// its own counts, so the elements advance through the rounds together, each
// round's evaluations running concurrently.
namespace {

// An element's share of a round of the arm search: the arms each point takes,
// the points it can still take, which have an arm that is starved now as
// counts only grow, and the element's values there.
struct RoundWork {
  SmallVector<SmallVector<RegimeArm, 4>, 0> arms;
  SmallVector<size_t, 0> wanted;
  SmallVector<double, 0> values;
  const SubgraphSamples::Reference *ref = nullptr;
};

} // namespace

// The work of E on the round's `points`, with its values computed ahead.
static RoundWork planRound(const Options &O, ElementPricing &E,
                           ArrayRef<SubgraphSamples::Point> points) {
  const size_t n = points.size();
  RoundWork W;
  W.arms.resize(n);
  forEachBlock(O.Model.Threads, n, 64, [&](size_t begin, size_t end) {
    for (size_t p = begin; p < end; ++p) {
      FPEvaluator ev;
      W.arms[p] = takenArms(E.node, points[p], ev);
    }
  });
  for (size_t p = 0; p < n; ++p)
    if (any_of(W.arms[p],
               [&](const RegimeArm &arm) { return E.starved(O, arm); }))
      W.wanted.push_back(p);
  W.values.resize(n);
  forEachBlock(O.Model.Threads, W.wanted.size(), 16,
               [&](size_t begin, size_t end) {
                 SmallVector<double, 1> r;
                 for (size_t i = begin; i < end; ++i) {
                   getFPValues({E.node}, points[W.wanted[i]], r);
                   W.values[W.wanted[i]] = r[0];
                 }
               });
  return W;
}

// Adds to E's starved arms the samples of the round `W`, in point order: a
// point counts only while one of its arms is still starved.
static void takeRound(const Options &O, ElementPricing &E, const RoundWork &W) {
  for (size_t p : W.wanted) {
    if (none_of(W.arms[p],
                [&](const RegimeArm &arm) { return E.starved(O, arm); }))
      continue;
    double origError = W.ref->error[p];
    double error = sampleError(W.ref->gold[p], W.values[p]);
    if (std::isnan(error) || std::isnan(origError))
      continue;
    for (const RegimeArm &arm : W.arms[p]) {
      if (!E.starved(O, arm))
        continue;
      ArmStats &st = E.armStats[arm];
      st.candSum += error;
      st.origSum += origError;
      ++st.count;
    }
  }
}

// Uniform marginal sampling weights each regime by its share of the profiled
// box, which need not be its share of the workload: correlated inputs can put
// most of the run in an arm the box almost never reaches. Every arm is priced
// on its own samples, drawn round by round until it has enough or the budget
// is spent. An element takes the rounds in order and at each point reads only
// its own counts, so the elements advance through the rounds together, each
// round's evaluations running concurrently.
static void searchArms(const Options &O, SubgraphSamples &samples,
                       ArrayRef<ElementPricing *> elements) {
  const size_t budget =
      static_cast<size_t>(O.Model.ArmSearchFactor) * O.Model.NumSamples;
  for (unsigned round = 1;; ++round) {
    SmallVector<ElementPricing *, 0> active;
    for (ElementPricing *E : elements)
      if (E->searching)
        active.push_back(E);
    if (active.empty())
      return;
    ArrayRef<SubgraphSamples::Point> points = samples.round(round);
    if (points.empty()) {
      for (ElementPricing *E : active)
        E->searching = false;
      return;
    }

    SmallVector<RoundWork, 0> work(active.size());
    forEachIndex(O.Model.Threads, active.size(),
                 [&](size_t a) { work[a] = planRound(O, *active[a], points); });
    // The exact values at the points the elements of one original take.
    std::map<const FPNode *, SmallVector<size_t, 0>> byOrig;
    for (auto [E, W] : zip(active, work))
      append_range(byOrig[E->orig], W.wanted);
    for (auto &[orig, which] : byOrig) {
      const SubgraphSamples::Reference &ref =
          samples.roundReference(orig, round, which);
      for (auto [E, W] : zip(active, work))
        if (E->orig == orig)
          W.ref = &ref;
    }

    forEachIndex(O.Model.Threads, active.size(), [&](size_t a) {
      ElementPricing &E = *active[a];
      E.drawn += points.size();
      takeRound(O, E, work[a]);
      E.searching = E.starved(O) && E.drawn < budget;
    });
  }
}

// Charges the element its worst arm's excess over the original.
static void chargeWorstArm(const Options &O, ElementPricing &E) {
  if (E.discard || E.ifNodes.empty())
    return;
  double worst = E.cost;
  std::string armReport;
  raw_string_ostream os(armReport);
  for (const FPNode *ifNode : E.ifNodes)
    for (bool taken : {false, true}) {
      const ArmStats &st = E.armStats[{ifNode, taken}];
      os << ' ' << (taken ? "then" : "else") << ':' << st.count;
      if (st.count)
        os << ':' << format("%.3e", st.candSum / st.count) << '/'
           << format("%.3e", st.origSum / st.count);
      if (st.count < O.Model.MinArmSamples) {
        E.discard = true;
        E.why = formatv("a regime-split arm is reached by fewer than {0} "
                        "samples",
                        (unsigned)O.Model.MinArmSamples)
                    .str();
        continue;
      }
      worst =
          std::max(worst, E.origCost + (st.candSum - st.origSum) / st.count);
    }
  if (O.Diagnostics.Print)
    raw_string_ostream(E.diag)
        << "[poseidon] regime-split candidate: arms (samples:cand/"
           "orig mean error)"
        << os.str() << "; " << E.drawn << " extra draws; box cost "
        << format("%.3e", E.cost) << " -> "
        << (E.discard ? std::string("dropped (unpriced arm)")
                      : formatv("{0:e}", worst).str())
        << "; " << E.exprForLog->substr(0, 120) << '\n';
  if (!E.discard)
    E.cost = worst;
}

static void collectInputs(const FPNode *node, const SetVector<Value *> &inputs,
                          SmallPtrSetImpl<const FPNode *> &seen,
                          SmallPtrSetImpl<Value *> &used) {
  if (!seen.insert(node).second)
    return;
  if (auto *LL = dyn_cast<FPLLValue>(node)) {
    used.insert(LL->value);
    if (inputs.contains(LL->value) && !inlinedOps(*LL, inputs))
      return;
  }
  for (const auto &operand : node->operands)
    collectInputs(operand, inputs, seen, used);
}

// The elements of a candidate of RU parsed as `parsed`, one per output: the
// operands of an array, or the candidate itself; none when an array candidate
// has another count.
static std::optional<SmallVector<FPNode *, 1>>
elementsOf(FPNode *parsed, const RewriteUnit &RU) {
  SmallVector<FPNode *, 1> elements;
  if (!RU.isArray()) {
    elements.push_back(parsed);
    return elements;
  }
  if (!parsed->is(fpcore::Array) || parsed->operands.size() != RU.roots.size())
    return std::nullopt;
  elements.append(parsed->operands.begin(), parsed->operands.end());
  return elements;
}

namespace {

// The accuracy model of one algebraic unit: each candidate's error on the
// subgraph's samples against the outputs as written, weighted by the outputs'
// gradients.
class RewriteAccuracy {
public:
  RewriteAccuracy(const Options &O, RewriteUnit &RU, FPGraph &G)
      : O(O), RU(RU), G(G), samples(samplesOf(O, *RU.subgraph, G)) {}

  // Sets the unit's accuracy cost as written and each candidate's, and drops
  // the candidates that cannot be priced.
  void price();

private:
  // One candidate's elements, priced against their outputs.
  struct CandidatePricing {
    FPNode *parsed;
    bool wrongShape = false;
    // The elements in order up to the first that reads a value its output
    // does not, whose index is `strayElement`.
    SmallVector<ElementPricing, 1> elements;
    std::optional<size_t> strayElement;
  };

  // `candidate` parsed into elements, each set up to be priced.
  CandidatePricing prepare(const RewriteCandidate &candidate) const;
  // Whether `candidate`, priced as `P`, is kept, with its accuracy cost set;
  // else why it is dropped is recorded.
  bool keep(RewriteCandidate &candidate, CandidatePricing &P);

  const Options &O;
  RewriteUnit &RU;
  FPGraph &G;
  SubgraphSamples &samples;
  SmallVector<BaselineAccuracy::Output, 1> rootSamples;
  // The values each output's original expression reads: an element is
  // spliced in at its own output, so it may read only those.
  SmallVector<SmallPtrSet<Value *, 8>, 1> rootInputs;
};

} // namespace

RewriteAccuracy::CandidatePricing
RewriteAccuracy::prepare(const RewriteCandidate &candidate) const {
  CandidatePricing P;
  P.parsed = cantFail(
      parseHerbieExpr(candidate.expr, G, nullptr, &RU.subgraph->inputs));
  std::optional<SmallVector<FPNode *, 1>> elements = elementsOf(P.parsed, RU);
  if (!elements) {
    P.wrongShape = true;
    return P;
  }
  for (auto [k, element] : enumerate(*elements)) {
    SmallPtrSet<const FPNode *, 32> seen;
    SmallPtrSet<Value *, 8> used;
    collectInputs(element, RU.subgraph->inputs, seen, used);
    const SmallPtrSet<Value *, 8> &elementInputs = rootInputs[k];
    if (!all_of(used, [&](Value *V) { return elementInputs.contains(V); })) {
      P.strayElement = k;
      break;
    }
    ElementPricing &E = P.elements.emplace_back();
    E.node = element;
    E.orig = rootSamples[k].node;
    E.ref = rootSamples[k].ref;
    E.origCost = rootSamples[k].error;
    E.exprForLog = &candidate.expr;
    if (O.Model.MinArmSamples > 0) {
      SmallPtrSet<const FPNode *, 32> seenIfs;
      collectIfNodes(element, seenIfs, E.ifNodes);
    }
    for (const FPNode *ifNode : E.ifNodes) {
      E.armStats[{ifNode, true}];
      E.armStats[{ifNode, false}];
    }
  }
  return P;
}

bool RewriteAccuracy::keep(RewriteCandidate &candidate, CandidatePricing &P) {
  if (P.wrongShape) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] dropping array candidate with "
                   << P.parsed->operands.size() << " element(s) for "
                   << RU.roots.size() << " outputs\n";
    RU.dropped.push_back(
        {candidate.expr, "it does not have one element per output"});
    return false;
  }

  double total = 0.0;
  std::string perElement;
  for (auto [k, E] : enumerate(P.elements)) {
    llvm::errs() << E.diag;
    if (E.discard) {
      RU.dropped.push_back({candidate.expr, E.why});
      return false;
    }
    const double grad = std::fabs(rootSamples[k].grad);
    total += E.cost * grad;
    if (RU.isArray() && O.Diagnostics.Print)
      perElement +=
          formatv(" {0:e}/{1:e}", E.cost * grad, rootSamples[k].error * grad)
              .str();
  }
  if (P.strayElement) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] dropping candidate: element "
                   << *P.strayElement << " reads a value its output does not\n";
    RU.dropped.push_back(
        {candidate.expr, formatv("its element {0} reads a value that "
                                 "output does not",
                                 *P.strayElement)
                             .str()});
    return false;
  }
  if (RU.isArray() && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] array candidate per element (cand/orig):"
                 << perElement << '\n';
  candidate.accuracyCost = total;
  assert(!std::isnan(candidate.accuracyCost) &&
         "a kept candidate has an accuracy cost");
  return true;
}

void RewriteAccuracy::price() {
  SmallVector<const FPNode *, 1> nodes;
  for (Value *root : RU.roots)
    nodes.push_back(G.valueToNodeMap[root]);
  std::optional<BaselineAccuracy> baseline =
      RU.priceBaseline(samples, nodes, RU.rootGrads, O.Diagnostics.Print);
  if (!baseline)
    return;
  ArrayRef<SubgraphSamples::Point> sampledPoints = baseline->points;
  rootSamples = std::move(baseline->outputs);
  for (Value *root : RU.roots) {
    SmallPtrSet<const FPNode *, 32> seen;
    SmallPtrSet<Value *, 8> used;
    collectInputs(G.valueToNodeMap[root], RU.subgraph->inputs, seen, used);
    rootInputs.push_back(std::move(used));
  }

  // Parsing extends the site's maps, so the candidates are read in order and
  // only their elements are priced concurrently.
  SmallVector<CandidatePricing, 4> pricings;
  for (const RewriteCandidate &candidate : RU.candidates)
    pricings.push_back(prepare(candidate));
  SmallVector<ElementPricing *, 0> allElements;
  for (CandidatePricing &P : pricings)
    for (ElementPricing &E : P.elements)
      allElements.push_back(&E);
  forEachIndex(O.Model.Threads, allElements.size(), [&](size_t i) {
    priceOnPoints(O, *allElements[i], sampledPoints);
  });
  searchArms(O, samples, allElements);
  for (ElementPricing *E : allElements)
    chargeWorstArm(O, *E);

  SmallVector<RewriteCandidate, 4> kept;
  for (auto [candidate, P] : zip(RU.candidates, pricings))
    if (keep(candidate, P))
      kept.push_back(std::move(candidate));
  RU.candidates = std::move(kept);
}

//===----------------------------------------------------------------------===//
// Cost of a rewrite
//===----------------------------------------------------------------------===//

namespace {

// A Herbie expression as a function of the values it reads, in a scratch
// function of the site's module that is erased with it.
class ScratchFunction {
public:
  ScratchFunction(Module &M, const std::string &expr, const FastMathFlags &FMF,
                  FPGraph &G, const SetVector<Value *> &subgraphInputs);
  ~ScratchFunction() { F->eraseFromParent(); }

  Function &function() const { return *F; }
  // The value argument A stands for.
  const Value *leafOf(const Argument &A) const { return leaves[A.getArgNo()]; }

private:
  Function *F = nullptr;
  // The values the expression reads: those its symbols name, then the
  // existing values its nodes were mapped back to.
  SmallVector<Value *, 8> leaves;
};

} // namespace

// The type the scratch function of a node of `dtype` over `argTypes` returns:
// its first floating-point (or vector of floating-point) argument's.
static Type *scratchReturnType(ArrayRef<Type *> argTypes, StringRef dtype,
                               LLVMContext &Ctx) {
  auto isFP = [](Type *T) {
    if (T->isFloatingPointTy())
      return true;
    auto *VT = dyn_cast<VectorType>(T);
    return VT && VT->getElementType()->isFloatingPointTy();
  };
  if (auto It = find_if(argTypes, isFP); It != argTypes.end())
    return *It;
  if (dtype == "f32")
    return Type::getFloatTy(Ctx);
  if (dtype == "f16")
    return Type::getHalfTy(Ctx);
  return Type::getDoubleTy(Ctx);
}

ScratchFunction::ScratchFunction(Module &M, const std::string &expr,
                                 const FastMathFlags &FMF, FPGraph &G,
                                 const SetVector<Value *> &subgraphInputs) {
  LLVMContext &Ctx = M.getContext();
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SetVector<Value *> args;
  SmallVector<Type *, 8> argTypes;
  for (const std::string &argStr : argStrSet) {
    Value *argValue = G.symbolToValueMap[argStr];
    args.insert(argValue);
    argTypes.push_back(argValue->getType());
    leaves.push_back(argValue);
  }

  FPNode *parsedNode =
      cantFail(parseHerbieExpr(expr, G, nullptr, &subgraphInputs));
  // An existing value a node was mapped back to is an argument no symbol names.
  SmallVector<FPNode *, 16> stack{parsedNode};
  SmallPtrSet<FPNode *, 16> seen;
  while (!stack.empty()) {
    FPNode *N = stack.pop_back_val();
    if (!seen.insert(N).second)
      continue;
    if (auto *L = dyn_cast<FPLLValue>(N)) {
      if (args.insert(L->value)) {
        argTypes.push_back(L->value->getType());
        leaves.push_back(L->value);
      }
      continue;
    }
    for (FPNode *op : N->operands)
      stack.push_back(op);
  }

  Type *ReturnType = scratchReturnType(
      argTypes, parsedNode->format ? parsedNode->format->Dtype : "", Ctx);
  // An array candidate returns all of its elements, so none is dead and the
  // subterms they share are simplified into one computation before costing.
  const bool isArray = parsedNode->is(fpcore::Array);
  Type *FuncRetType = ReturnType;
  if (isArray)
    FuncRetType = StructType::get(
        Ctx, SmallVector<Type *>(parsedNode->operands.size(), ReturnType));

  FunctionType *FT = FunctionType::get(FuncRetType, argTypes, false);
  F = Function::Create(FT, Function::InternalLinkage, "tempFunc", &M);

  ValueToValueMapTy VMap;
  for (auto [leaf, arg] : zip(leaves, F->args()))
    VMap[leaf] = &arg;

  IRBuilder<> builder(BasicBlock::Create(Ctx, "entry", F));
  builder.setFastMathFlags(FMF);
  if (isArray) {
    SmallVector<Value *, 4> elements = emitSharedElements(
        G, parsedNode->operands,
        [&](size_t) -> IRBuilder<> & { return builder; }, &VMap, nullptr);
    Value *Agg = PoisonValue::get(FuncRetType);
    for (auto [i, V] : enumerate(elements)) {
      if (V->getType() != ReturnType)
        V = builder.CreateFPCast(V, ReturnType);
      Agg = builder.CreateInsertValue(Agg, V, i);
    }
    builder.CreateRet(Agg);
  } else {
    Value *RetVal = parsedNode->getLLValue(builder, &VMap);
    assert(RetVal && "Parsed node did not produce a value");
    if (RetVal->getType() != ReturnType)
      RetVal = builder.CreateFPCast(RetVal, ReturnType);
    builder.CreateRet(RetVal);
  }
}

namespace {

// What each instruction of a rewrite's scratch function executes per
// execution of the rewritten outputs: a conversion of an argument or of a
// returned value is on the boundary of the rewrite and costs what it executes
// in the emitted code, as `Boundary` says; everything else once.
class BoundaryWeights {
public:
  BoundaryWeights(const ScratchFunction &Scratch,
                  const BoundaryCastFreq &Boundary);
  double operator()(const Instruction &I) const;

private:
  const ScratchFunction &Scratch;
  const BoundaryCastFreq &Boundary;
  SmallPtrSet<const Value *, 4> returned;
  // Whether the loop carries the rewrite's values converted: it reads the
  // carried values only converted and returns only converted results.
  bool carriesConverted = false;
};

} // namespace

BoundaryWeights::BoundaryWeights(const ScratchFunction &Scratch,
                                 const BoundaryCastFreq &Boundary)
    : Scratch(Scratch), Boundary(Boundary) {
  Function &F = Scratch.function();
  carriesConverted = !Boundary.carried.empty();
  for (BasicBlock &BB : F)
    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      Value *RV = RI->getReturnValue();
      while (auto *IV = dyn_cast_or_null<InsertValueInst>(RV)) {
        returned.insert(IV->getInsertedValueOperand());
        RV = IV->getAggregateOperand();
      }
      if (RV && !isa<UndefValue>(RV))
        returned.insert(RV);
    }
  for (const Value *RV : returned)
    carriesConverted &= isa<FPExtInst, FPTruncInst>(RV);
  for (const Argument &A : F.args())
    if (Boundary.carried.count(Scratch.leafOf(A)))
      for (const User *U : A.users())
        carriesConverted &= isa<FPExtInst, FPTruncInst>(U);
}

double BoundaryWeights::operator()(const Instruction &I) const {
  if (!isa<FPExtInst, FPTruncInst>(I))
    return 1.0;
  if (auto *A = dyn_cast<Argument>(I.getOperand(0))) {
    const Value *Leaf = Scratch.leafOf(*A);
    if (auto It = Boundary.hoisted.find(Leaf); It != Boundary.hoisted.end())
      return It->second;
    if (auto It = Boundary.carried.find(Leaf);
        carriesConverted && It != Boundary.carried.end())
      return It->second;
    return 1.0;
  }
  return carriesConverted && returned.contains(&I) ? Boundary.output : 1.0;
}

// The cost of the Herbie expression `expr` that replaces `Site`, priced in a
// scratch function of its module simplified by `PB`'s pipeline; a missing
// cost-model row is an error at `Site`.
static Expected<double> priceRewrite(CostModelState &CM, PassBuilder &PB,
                                     const std::string &expr, Instruction &Site,
                                     FPGraph &G, const FastMathFlags &FMF,
                                     const SetVector<Value *> &subgraphInputs,
                                     const BoundaryCastFreq &Boundary) {
  ScratchFunction Scratch(*Site.getModule(), expr, FMF, G, subgraphInputs);
  simplifyFunction(Scratch.function(), PB, OptimizationLevel::O3);
  BoundaryWeights weight(Scratch, Boundary);
  return compCost(CM, &Scratch.function(), weight, &Site);
}

BoundaryCastFreq RewriteUnit::boundaryCastFreq(const LoopInfo &LI) const {
  const auto &valueToNodeMap = graph().valueToNodeMap;
  BoundaryCastFreq B;
  const Loop *L = LI.getLoopFor(cast<Instruction>(roots.front())->getParent());
  if (!L || executions == 0)
    return B;
  for (Value *Root : roots)
    if (LI.getLoopFor(cast<Instruction>(Root)->getParent()) != L)
      return B;

  // The measured executions of a value computed outside the loop, per
  // execution of the outputs.
  auto outside = [&](const Value *V) -> std::optional<double> {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || L->contains(I))
      return std::nullopt;
    auto It = valueToNodeMap.find(const_cast<Value *>(V));
    if (It == valueToNodeMap.end() || !It->second || !It->second->executions)
      return std::nullopt;
    return std::min(1.0, (double)It->second->executions / (double)executions);
  };

  double carried = 0.0;
  SmallPtrSet<const PHINode *, 2> carriedPHIs;
  for (Value *In : subgraph->inputs) {
    if (std::optional<double> Freq = outside(In)) {
      B.hoisted[In] = *Freq;
      continue;
    }
    // A header PHI whose value on every back edge is an output of this rewrite
    // and that nothing else in the loop reads: the loop carries the value in
    // the rewrite's precision, converted where the loop is entered.
    auto *PN = dyn_cast<PHINode>(In);
    if (!PN || PN->getParent() != L->getHeader())
      continue;
    double entry = 0.0;
    bool isCarried = true;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *V = PN->getIncomingValue(i);
      if (L->contains(PN->getIncomingBlock(i)))
        isCarried &= is_contained(roots, V);
      else if (std::optional<double> Freq = outside(V))
        entry += *Freq;
      else
        isCarried &= isa<Constant>(V);
    }
    for (const User *U : PN->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      isCarried &=
          UI && (!L->contains(UI) ||
                 erasableInsts.contains(const_cast<Instruction *>(UI)));
    }
    if (!isCarried || entry == 0.0)
      continue;
    B.carried[In] = std::min(1.0, entry);
    carried = std::max(carried, B.carried[In]);
    carriedPHIs.insert(PN);
  }

  // The outputs are converted back only where the loop is left when every
  // reader inside it is such a PHI; otherwise nothing is carried converted.
  bool leavesConverted = !carriedPHIs.empty();
  for (Value *Root : roots)
    for (const User *U : Root->users()) {
      auto *UI = dyn_cast<Instruction>(U);
      if (UI && (!L->contains(UI) ||
                 erasableInsts.contains(const_cast<Instruction *>(UI))))
        continue;
      auto *PN = dyn_cast_or_null<PHINode>(UI);
      leavesConverted &= PN && carriedPHIs.contains(PN);
    }
  if (leavesConverted)
    B.output = carried;
  else
    B.carried.clear();
  return B;
}

//===----------------------------------------------------------------------===//
// The unit
//===----------------------------------------------------------------------===//

Error RewriteUnit::materialize(size_t candidateIndex) {
  FPGraph &G = graph();
  auto &valueToNodeMap = G.valueToNodeMap;
  assert(candidateIndex < candidates.size() &&
         "the solver names one of the candidates");
  assert(all_of(roots,
                [](Value *root) {
                  return cast<Instruction>(root)->getParent();
                }) &&
         "a rewrite's outputs are in the site when it is materialized");

  FPNode *parsedNode = cantFail(parseHerbieExpr(candidates[candidateIndex].expr,
                                                G, nullptr, &subgraph->inputs));
  std::optional<SmallVector<FPNode *, 1>> elements =
      elementsOf(parsedNode, *this);
  if (!elements)
    report_fatal_error("Poseidon: an array rewrite does not have one element "
                       "per output of its subgraph");

  SmallVector<Value *, 4> newOutputs;
  if (isArray()) {
    DominatorTree DT(*cast<Instruction>(roots.front())->getFunction());
    std::optional<IRBuilder<>> builder;
    newOutputs = emitSharedElements(
        G, *elements,
        [&](size_t k) -> IRBuilder<> & {
          auto *oldInst = cast<Instruction>(roots[k]);
          builder.emplace(oldInst->getParent());
          setInsertPoint(*builder, oldInst->getParent(),
                         ++BasicBlock::iterator(oldInst));
          builder->setFastMathFlags(oldInst->getFastMathFlags());
          return *builder;
        },
        nullptr, &DT);
  }

  for (auto [k, root, element] : llvm::enumerate(roots, *elements)) {
    auto *oldInst = cast<Instruction>(root);
    IRBuilder<> builder =
        builderAt(oldInst->getParent(), ++BasicBlock::iterator(oldInst));
    builder.setFastMathFlags(oldInst->getFastMathFlags());

    Value *newOutput = isArray() ? newOutputs[k] : element->getLLValue(builder);
    assert(newOutput && "Failed to get value from parsed node");
    if (isArray() && isa<Instruction>(newOutput))
      builder.SetInsertPoint(cast<Instruction>(newOutput)->getNextNode());

    if (newOutput->getType() != root->getType())
      newOutput =
          builder.CreateFPCast(newOutput, root->getType(), "herbie.fpcast");

    root->replaceAllUsesWith(newOutput);
    G.symbolToValueMap[valueToNodeMap[root]->symbol] = newOutput;
    valueToNodeMap[newOutput] =
        G.create<FPLLValue>(newOutput, nullptr, valueToNodeMap[root]->format);
  }

  for (Instruction *I : erasableInsts) {
    if (!I->use_empty())
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    I->eraseFromParent();
    subgraph->operations.remove(I);
    cast<FPLLValue>(valueToNodeMap[I])->value = nullptr;
  }

  subgraph->outputsRewritten += roots.size();
  return Error::success();
}

Error RewriteUnit::priceAsWritten(CostModelState &CM) {
  Expected<double> cost = compCost(CM, roots, subgraph->inputs);
  if (!cost)
    return cost.takeError();
  initialCompCost = *cost;
  erasableCost = 0.0;
  for (Instruction *I : erasableInsts) {
    Expected<double> price = instructionCost(CM, I);
    if (!price)
      return price.takeError();
    erasableCost += *price;
  }
  return Error::success();
}

double RewriteUnit::getCompCostDelta(size_t candidateIndex) {
  return (candidates[candidateIndex].CompCost - erasableCost) *
         (double)executions;
}

// The instructions `V` is computed by, back to `inputs`.
static void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
                             SmallPtrSetImpl<Instruction *> &exprInsts,
                             SmallPtrSetImpl<Value *> &visited) {
  if (!V || inputs.contains(V) || !visited.insert(V).second)
    return;
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return;
  exprInsts.insert(I);
  for (Value *op : isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands())
    collectExprInsts(op, inputs, exprInsts, visited);
}

void RewriteUnit::findErasableInstructions() {
  SmallPtrSet<Value *, 8> visited;
  SmallPtrSet<Instruction *, 8> exprInsts;
  for (Value *root : roots)
    collectExprInsts(root, subgraph->inputs, exprInsts, visited);
  visited.clear();

  // Seeded in IR order, not in the order the pointers happened to hash: the
  // topological sort below is stable, so its input order decides the order the
  // report lists these instructions in and the order their costs are summed.
  SetVector<Instruction *> instsToProcess;
  for (Instruction &I :
       instructions(*cast<Instruction>(roots.front())->getFunction()))
    if (exprInsts.contains(&I))
      instsToProcess.insert(&I);

  SmallVector<Instruction *, 8> instsToProcessSorted;
  topoSort(instsToProcess, instsToProcessSorted);
  std::reverse(instsToProcessSorted.begin(), instsToProcessSorted.end());

  erasableInsts.clear();
  for (Value *root : roots)
    erasableInsts.insert(cast<Instruction>(root));

  for (Instruction *I : instsToProcessSorted)
    if (all_of(I->users(), [&](User *U) {
          auto *UI = dyn_cast<Instruction>(U);
          return UI && erasableInsts.contains(UI);
        }))
      erasableInsts.insert(I);
}

double RewriteUnit::costDelta(size_t i, ArrayRef<SolutionStep>) {
  return getCompCostDelta(i);
}

double RewriteUnit::errorDelta(size_t i, ArrayRef<SolutionStep>) {
  return getAccCostDelta(i);
}

bool RewriteUnit::sameRewriteTargetAs(const OptimizationUnit &earlier) const {
  const auto *other = dyn_cast<RewriteUnit>(&earlier);
  return other && other->subgraph == subgraph &&
         other->roots.size() == roots.size() &&
         llvm::all_of(other->roots,
                      [&](Value *r) { return llvm::is_contained(roots, r); });
}

std::string RewriteUnit::choiceText(size_t i) const {
  const RewriteCandidate &cand = candidates[i];
  return expr + " --(" + std::to_string(cand.ordinal) + ")-> " + cand.expr;
}

Error RewriteUnit::price() {
  if (!herbieAnswered)
    return Error::success();
  FPGraph &st = graph();
  auto *output = cast<Instruction>(roots.front());
  const FastMathFlags FMF = output->getFastMathFlags();
  DominatorTree DT(*st.F);
  LoopInfo LI(DT);
  const BoundaryCastFreq Boundary = boundaryCastFreq(LI);
  for (RewriteCandidate &c : candidates) {
    Expected<double> cost =
        priceRewrite(session().CostModel, session().PB, c.expr, *output, st,
                     FMF, subgraph->inputs, Boundary);
    if (!cost)
      return cost.takeError();
    c.CompCost = *cost;
  }
  RewriteAccuracy(session().Opts, *this, st).price();
  return Error::success();
}

void RewriteUnit::printPriced(raw_ostream &OS) {
  OS << "[poseidon] priced rewrite candidates\n";
  OS << "Initial AccuracyCost: " << initialAccCost << '\n';
  OS << "Initial ComputationCost: " << initialCompCost << '\n';
  OS << "Initial HerbieAccuracy: " << initialHerbieAccuracy << '\n';
  OS << "Initial Expression: " << expr << '\n';
  // One output's gradient; for an array, the sum of its outputs' magnitudes.
  double grad = rootGrads.front();
  if (rootGrads.size() > 1) {
    grad = 0.0;
    for (double g : rootGrads)
      grad += std::fabs(g);
  }
  OS << "Grad: " << grad << "\n\n";
  OS << "Candidates:\n";
  OS << "accuracy delta\t\tcost delta\t\tHerbie cost\t\taccuracy\t\t"
        "expression\n";
  OS << "--------------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    RewriteCandidate &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t";
    // A compile that reads its DP table from the cache prices no candidate.
    if (candidate.CompCost == std::numeric_limits<double>::max())
      OS << "unpriced";
    else
      OS << getCompCostDelta(i);
    OS << "\t\t" << candidate.herbieCost << "\t\t" << candidate.herbieAccuracy
       << "\t\t" << candidate.expr << '\n';
  }
}

} // namespace poseidon
