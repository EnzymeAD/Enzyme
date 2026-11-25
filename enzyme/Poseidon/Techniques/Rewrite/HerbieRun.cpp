//===- HerbieRun.cpp - Running Herbie, with its result cache --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HerbieRun.h"
#include "../../Support/CacheKey.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <optional>

using namespace llvm;

namespace poseidon {

// `Name` in the directory `Dir`.
static std::string inDirectory(StringRef Dir, const Twine &Name) {
  SmallString<256> Path(Dir);
  sys::path::append(Path, Name);
  return std::string(Path);
}

std::string readFileDigest(StringRef Path) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    return std::string();
  return contentDigest((*Buf)->getBuffer());
}

// The digest of the Herbie binary at `Program`, read once per compile.
//
// A Herbie cache entry is stamped with the digest of the binary that produced
// it: version strings do not change between upstream commits, and one
// binary's rewrites must not be replayed as another's.
static std::string herbieBinaryDigest(StringRef Program,
                                      HerbieBinaryDigests &Digests) {
  auto It = Digests.find(Program.str());
  if (It == Digests.end())
    It = Digests.emplace(Program.str(), readFileDigest(Program)).first;
  return It->second;
}

// The arguments that decide what Herbie returns for a core. Its timeouts and
// worker count decide only whether and how soon it returns.
static SmallVector<std::string> herbieSearchArgs(const HerbieRunOptions &Opts) {
  // Herbie prunes alternatives by tree cost unless told otherwise, paying a
  // shared subterm once per occurrence; the emitted code computes it once.
  SmallVector<std::string> Args = {
      "--seed",       std::to_string(Opts.Seed),
      "--num-points", std::to_string(Opts.NumPoints),
      "--num-iters",  std::to_string(Opts.NumIters),
      "--num-enodes", std::to_string(Opts.NumEnodes),
      "--disable",    "generate:proofs",
      "--enable",     "reduce:dag-cost"};
  if (Opts.TaylorTerms)
    Args.append({"--num-taylor-terms", std::to_string(*Opts.TaylorTerms)});
  return Args;
}

// Whether `herbie report --help` of the Herbie at `Program` lists
// --num-taylor-terms.
static bool takesTaylorTerms(StringRef Program) {
  SmallString<128> Help;
  if (sys::fs::createTemporaryFile("herbie_help", "txt", Help))
    return false;
  FileRemover RemoveHelp(Help);
  std::optional<StringRef> Redirects[] = {std::nullopt, StringRef(Help),
                                          StringRef(Help)};
  sys::ExecuteAndWait(Program, {Program, "report", "--help"},
                      /*Env=*/std::nullopt, Redirects, /*SecondsToWait=*/60);
  auto Buf = MemoryBuffer::getFile(Help);
  return Buf && (*Buf)->getBuffer().contains("--num-taylor-terms");
}

// Writes the entry `value`, answered by the Herbie `answeredBy` names, to the
// cache file `path`, creating the cache directory.
static Error writeAnswer(StringRef cacheDir, StringRef path,
                         const CacheKey &answeredBy, json::Value value) {
  static constexpr StringLiteral kWritable =
      ". Herbie's answers are kept in the -poseidon-cache directory, which has "
      "to be writable.";
  if (std::error_code EC = sys::fs::create_directories(cacheDir, true))
    return createStringError(EC, "cannot create the Herbie cache directory " +
                                     cacheDir + ": " + EC.message() +
                                     kWritable);
  if (Error E = writeCacheEntry(path, answeredBy, std::move(value)))
    return createStringError(inconvertibleErrorCode(),
                             "cannot write the Herbie cache file " + path +
                                 ": " + toString(std::move(E)) + kWritable);
  return Error::success();
}

namespace {
// How the cores are split across processes and what the run may spend. Herbie
// runs a process's cores across its workers, so ceil(cores/threads) rounds of
// the per-core timeout is what it can legitimately spend; the hard kill sits
// one margin above that. A wall budget of 0 waits for as long as Herbie takes.
struct ShardPlan {
  unsigned shards, threadsPerShard, coreTimeout, wallBudget;
};

// Where a question's answer and its timeout marker are kept, and which Herbie
// an answer has to come from to be served.
struct AnswerCache {
  std::string answerPath, timeoutPath;
  CacheKey answeredBy;
  // Without a Herbie to run, an answer any Herbie gave is the answer.
  bool haveHerbie;
  SmallVector<StringRef, 1> unchecked;
};

// One input file and one output directory per shard, removed with it.
class ShardFiles {
public:
  ShardFiles(ArrayRef<std::string> cores, unsigned shards);
  ~ShardFiles();
  bool created() const { return ok; }
  SmallVector<SmallString<64>> inputs, outputs;

private:
  bool ok = true;
};

// How the processes of a run ended.
struct ProcessOutcome {
  bool executionFailed = false;
  std::string executionError;
  bool killed = false;
  // The signal that ended a process the runner did not kill.
  std::string crash;
  // The first nonzero exit status.
  int status = 0;
  double seconds = 0;
};
} // namespace

static ShardPlan planShards(size_t numCores, const HerbieRunOptions &Opts) {
  ShardPlan plan;
  const unsigned numThreads = std::max(1, Opts.NumThreads);
  plan.shards = (unsigned)std::max<size_t>(
      1,
      std::min<size_t>(Opts.Processes ? Opts.Processes : numThreads, numCores));
  plan.threadsPerShard = std::max<unsigned>(
      1, std::min<size_t>((numThreads + plan.shards - 1) / plan.shards,
                          (numCores + plan.shards - 1) / plan.shards));
  plan.coreTimeout = (unsigned)Opts.CoreTimeout;
  plan.wallBudget = 0;
  if (Opts.SubgraphTimeout == 0)
    return plan;
  plan.coreTimeout =
      std::min<unsigned>((unsigned)Opts.CoreTimeout, Opts.SubgraphTimeout);
  unsigned perShard = (unsigned)((numCores + plan.shards - 1) / plan.shards);
  unsigned rounds = std::max<unsigned>(
      1, (perShard + plan.threadsPerShard - 1) / plan.threadsPerShard);
  plan.wallBudget = plan.coreTimeout * rounds + plan.coreTimeout / 10 + 60;
  if (Opts.Print && plan.coreTimeout < (unsigned)Opts.CoreTimeout)
    llvm::errs() << "[poseidon] Herbie per-core --timeout clamped from "
                 << Opts.CoreTimeout << " s to " << plan.coreTimeout
                 << " s by -poseidon-herbie-subgraph-timeout\n";
  return plan;
}

// The command line of one shard, without its input file and output directory.
static SmallVector<std::string> herbieCommand(const HerbieRunOptions &Opts,
                                              const ShardPlan &plan) {
  const HerbiePlatform &Platform = Opts.Platform;
  SmallVector<std::string> args = {
      Opts.Program, "report",
      "--timeout",  std::to_string(plan.coreTimeout),
      "--threads",  std::to_string(plan.threadsPerShard)};
  args.append(herbieSearchArgs(Opts));
  // "default" is Herbie's own choice of platform, which it makes when it is
  // not told one; a host cost model leaves it there.
  if (Platform.Name == "default" && Platform.Path.empty())
    return args;
  args.push_back("--platform");
  args.push_back(Platform.Path.empty() ? Platform.Name : Platform.Path);
  if (Opts.Print)
    llvm::errs() << "[poseidon] using Herbie platform '" << Platform.Name
                 << '\''
                 << (Platform.Path.empty()
                         ? std::string()
                         : " from " + Platform.Path + " (digest " +
                               Platform.Digest + ")")
                 << '\n';
  return args;
}

// The digest of the question: the cores, the search arguments and the
// platform Herbie ranks candidates by.
static std::string questionDigest(ArrayRef<std::string> Cores,
                                  const HerbieRunOptions &Opts) {
  std::string joinedCores;
  for (const std::string &expr : Cores) {
    joinedCores += expr;
    joinedCores += '\n';
  }
  CacheKey question;
  question.add("Herbie platform",
               Opts.Platform.Name + " " + Opts.Platform.Digest);
  question.add("Herbie arguments", join(herbieSearchArgs(Opts), " "));
  question.addContent("cores", joinedCores);
  return question.digest();
}

// The answer, or the timeout, the cache holds for the question, a run's
// result; none when Herbie has to run. Without a Herbie, a question the cache
// does not answer is an error naming its cache file.
static Expected<std::optional<HerbieRunResult>>
readCachedAnswer(const AnswerCache &cache, const HerbieRunOptions &Opts,
                 StringRef Label, unsigned wallBudget) {
  HerbieRunResult Run;
  CacheEntryRead cached =
      readCacheEntry(cache.answerPath, cache.answeredBy, cache.unchecked);
  if (cached.value) {
    raw_string_ostream(Run.ResultsJson) << *cached.value;
    if (Opts.Print)
      llvm::errs() << "[poseidon] Using cached Herbie output from "
                   << cache.answerPath
                   << "\n[poseidon] Herbie output: " << Run.ResultsJson << '\n';
    Run.Result = HerbieRunResult::Outcome::Ok;
    Run.From = HerbieRunResult::Origin::Cache;
    Run.CacheFile = cache.answerPath;
    return Run;
  }
  if (!cached.unreadable.empty() && !cache.haveHerbie)
    return createStringError(inconvertibleErrorCode(),
                             "the Herbie cache file " + cache.answerPath +
                                 " for " + Label + " is not a cache entry (" +
                                 cached.unreadable + ")");
  if (Opts.Print && (!cached.unreadable.empty() || !cached.differs.empty()))
    llvm::errs() << "[poseidon] Herbie " << Label << ": the cache file "
                 << cache.answerPath << " holds "
                 << (cached.unreadable.empty()
                         ? "the answer of another Herbie"
                         : "no cache entry (" + cached.unreadable + ")")
                 << "; running Herbie to replace it\n";

  // A recorded timeout is a result too: without it every re-solve pays the
  // full wall-clock budget again to learn the same thing.
  CacheEntryRead marker =
      readCacheEntry(cache.timeoutPath, cache.answeredBy, cache.unchecked);
  std::optional<uint64_t> spentBudget =
      marker.value ? marker.value->getAsUINT64() : std::nullopt;
  if (spentBudget && wallBudget > 0 && *spentBudget >= wallBudget) {
    Run.Result = HerbieRunResult::Outcome::Timeout;
    Run.Warning = ("Herbie " + Label + " timed out before on this exact " +
                   "input with a budget of " + Twine(*spentBudget) +
                   " s (recorded in " + cache.timeoutPath +
                   "), so the subgraph has no algebraic candidates; raise "
                   "-poseidon-herbie-subgraph-timeout above " +
                   Twine(*spentBudget) + " to retry it")
                      .str();
    return Run;
  }
  if (!cache.haveHerbie)
    return createStringError(
        inconvertibleErrorCode(),
        "no Herbie answer for " + Label + " is cached in " + cache.answerPath +
            ", and " +
            (Opts.Program.empty()
                 ? kBuiltWithoutHerbie.str()
                 : "there is no Herbie at " + Opts.Program + " to compute it") +
            ". Compute the answer on a machine with Herbie and copy its "
            "-poseidon-cache directory here, or turn algebraic rewriting off "
            "with -poseidon-enable-rewrite=0.");
  return std::nullopt;
}

ShardFiles::ShardFiles(ArrayRef<std::string> cores, unsigned shards)
    : inputs(shards), outputs(shards) {
  for (unsigned s = 0; s < shards && ok; ++s) {
    if (sys::fs::createUniqueFile("herbie_input_%%%%%%%%%%%%%%%%", inputs[s],
                                  sys::fs::perms::owner_all) ||
        sys::fs::createUniqueDirectory("herbie_output_%%%%%%%%%%%%%%%%",
                                       outputs[s])) {
      ok = false;
      break;
    }
    std::ofstream input(inputs[s].c_str());
    for (size_t k = s; k < cores.size(); k += shards)
      input << cores[k] << "\n";
    input.close();
    if (!input)
      ok = false;
  }
}

ShardFiles::~ShardFiles() {
  for (unsigned s = 0; s < inputs.size(); ++s) {
    if (!inputs[s].empty())
      sys::fs::remove(inputs[s]);
    // A temporary directory left behind is harmless; llvm::FileRemover
    // ignores the same failure.
    if (!outputs[s].empty())
      sys::fs::remove_directories(outputs[s]);
  }
}

// Starts one process per shard and waits for them within the wall budget,
// killing what outlives it.
static ProcessOutcome runShards(const ShardFiles &files,
                                ArrayRef<std::string> command,
                                StringRef Program, unsigned wallBudget) {
  ProcessOutcome out;
  auto start = std::chrono::steady_clock::now();
  auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
  };
  SmallVector<sys::ProcessInfo> procs;
  for (unsigned s = 0; s < files.inputs.size() && !out.executionFailed; ++s) {
    SmallVector<StringRef> args(command.begin(), command.end());
    args.push_back(files.inputs[s]);
    args.push_back(files.outputs[s]);
    sys::ProcessInfo PI = sys::ExecuteNoWait(
        Program, args, /*Env=*/std::nullopt, /*Redirects=*/{},
        /*MemoryLimit=*/0, &out.executionError, &out.executionFailed);
    if (!out.executionFailed)
      procs.push_back(PI);
  }
  for (sys::ProcessInfo &PI : procs) {
    std::optional<unsigned> waitFor;
    if (out.executionFailed || out.killed || !out.crash.empty()) {
      waitFor = 1;
    } else if (wallBudget) {
      double spent = elapsed();
      waitFor = spent + 1 >= wallBudget ? 1u : (unsigned)(wallBudget - spent);
    }
    std::string waitError;
    sys::ProcessInfo R = sys::Wait(PI, waitFor, &waitError);
    // Wait returns -2 both when its own alarm killed the process and when a
    // signal from elsewhere (the OOM killer's SIGKILL) ended it; only its
    // message tells the two apart.
    if (R.ReturnCode == -2) {
      if (StringRef(waitError).starts_with("Child timed out"))
        out.killed = true;
      else if (out.crash.empty())
        out.crash = "signal: " + waitError;
    } else if (R.ReturnCode != 0 && out.status == 0) {
      out.status = R.ReturnCode;
    }
  }
  out.seconds = elapsed();
  return out;
}

// The results.json of the shards, merged: one process's verbatim into
// `content`, several processes' tests appended to the first one's report.
// Returns none when a shard wrote no usable results.json.
static std::optional<json::Value> readShardResults(const ShardFiles &files,
                                                   std::string &content) {
  std::optional<json::Value> answer;
  for (const SmallString<64> &output : files.outputs) {
    auto Buf = MemoryBuffer::getFile(inDirectory(output, "results.json"));
    if (!Buf)
      return std::nullopt;
    Expected<json::Value> parsed = json::parse((*Buf)->getBuffer());
    if (!parsed) {
      consumeError(parsed.takeError());
      return std::nullopt;
    }
    if (!parsed->getAsObject() || !parsed->getAsObject()->getArray("tests"))
      return std::nullopt;
    if (!answer) {
      if (files.outputs.size() == 1)
        content = (*Buf)->getBuffer().str();
      answer = std::move(*parsed);
      continue;
    }
    json::Array *dst = answer->getAsObject()->getArray("tests");
    for (json::Value &t : *parsed->getAsObject()->getArray("tests"))
      dst->push_back(std::move(t));
  }
  if (files.outputs.size() > 1) {
    content.clear();
    raw_string_ostream(content) << *answer;
  }
  return answer;
}

// `v` as raw_ostream prints a double.
static std::string printed(double v) {
  std::string text;
  raw_string_ostream(text) << v;
  return text;
}

// Why a run that wrote no usable results.json failed.
static std::string noResultsReason(const HerbieRunOptions &Opts, int status) {
  if (Opts.TaylorTerms && !takesTaylorTerms(Opts.Program))
    return "-poseidon-herbie-num-taylor-terms passes it --num-taylor-terms, "
           "which it does not take ('report --help' does not list it); use a "
           "Herbie built with the option, or leave the flag out";
  std::string where = Opts.Platform.Name;
  if (!Opts.Platform.Path.empty())
    where += ", " + Opts.Platform.Path;
  return (Twine("it exited with status ") + Twine(status) +
          " and wrote no usable results.json (platform " + where +
          "); its own diagnostics are on stderr above")
      .str();
}

// Where the answer to the question `keyDigest` is filed, the platform name
// leading the file name so that one device's answers are told apart from
// another's at a glance; the entry records the Herbie that gave it.
static AnswerCache answerCache(const HerbieRunOptions &Opts,
                               StringRef keyDigest, StringRef binaryDigest) {
  AnswerCache cache;
  cache.answeredBy.add("Herbie binary", binaryDigest);
  cache.haveHerbie = !binaryDigest.empty();
  if (!cache.haveHerbie)
    cache.unchecked.push_back("Herbie binary");
  if (Opts.CacheDir.empty())
    return cache;
  std::string keyStem = Opts.Platform.Name + "_" + keyDigest.str();
  cache.answerPath = inDirectory(Opts.CacheDir, "herbie_" + keyStem + ".json");
  // A timeout gets its own marker so it never looks like an empty result.
  cache.timeoutPath =
      inDirectory(Opts.CacheDir, "herbie_timeout_" + keyStem + ".txt");
  return cache;
}

static void announceRun(const HerbieRunOptions &Opts, StringRef Label,
                        size_t numCores, const ShardPlan &plan,
                        ArrayRef<std::string> command, StringRef keyDigest) {
  if (!Opts.Print)
    return;
  llvm::errs() << "[poseidon] Executing Herbie (" << plan.shards
               << " process(es), one input file and output directory "
                  "each) with arguments: ";
  for (const std::string &arg : command)
    llvm::errs() << arg << ' ';
  llvm::errs() << '\n';
  llvm::errs() << "[poseidon] Herbie " << Label << ": " << numCores
               << " FPCore(s) in " << plan.shards << " process(es) of "
               << plan.threadsPerShard << " worker(s), per-core timeout "
               << plan.coreTimeout << " s, wall budget "
               << (plan.wallBudget ? std::to_string(plan.wallBudget) + " s"
                                   : std::string("unbounded"))
               << ", key " << keyDigest << '\n';
}

// The result of a run whose processes were killed at the wall budget or ended
// by a signal; none for a run that ran to its end. A timeout is recorded in
// the cache, as it is a property of the input; a crash is not, so the next
// compile runs Herbie again.
static Expected<std::optional<HerbieRunResult>>
stoppedRun(const ProcessOutcome &ran, const AnswerCache &cache,
           const HerbieRunOptions &Opts, StringRef Label, unsigned wallBudget) {
  HerbieRunResult Run;
  if (!ran.crash.empty()) {
    Run.Result = HerbieRunResult::Outcome::Crashed;
    Run.Crash = ran.crash;
    Run.Warning = ("Herbie " + Label + " crashed (" + ran.crash + ") after " +
                   printed(ran.seconds) +
                   " s, so the subgraph has no algebraic candidates in this "
                   "compile; nothing is cached, so the next compile runs "
                   "Herbie again")
                      .str();
    return Run;
  }
  if (!ran.killed)
    return std::nullopt;
  Run.Result = HerbieRunResult::Outcome::Timeout;
  Run.Warning =
      ("Herbie " + Label + " was stopped after " + printed(ran.seconds) +
       " s, its wall budget of " + Twine(wallBudget) +
       " s, so the subgraph has no algebraic candidates; raise "
       "-poseidon-herbie-subgraph-timeout to spend more, or set it "
       "to 0 to wait for as long as Herbie takes")
          .str();
  if (!cache.timeoutPath.empty()) {
    if (Error E = writeAnswer(Opts.CacheDir, cache.timeoutPath,
                              cache.answeredBy, (uint64_t)wallBudget))
      return std::move(E);
    Run.Warning += "; the timeout is recorded in " + cache.timeoutPath;
  }
  return Run;
}

Expected<HerbieRunResult> runHerbie(const std::vector<std::string> &Cores,
                                    const HerbieRunOptions &Opts,
                                    StringRef Label,
                                    HerbieBinaryDigests &Digests) {
  const ShardPlan plan = planShards(Cores.size(), Opts);
  const SmallVector<std::string> command = herbieCommand(Opts, plan);
  const std::string keyDigest = questionDigest(Cores, Opts);
  const std::string binaryDigest =
      Opts.CacheOnly ? std::string()
                     : herbieBinaryDigest(Opts.Program, Digests);
  const AnswerCache cache = answerCache(Opts, keyDigest, binaryDigest);
  if (!Opts.CacheDir.empty()) {
    Expected<std::optional<HerbieRunResult>> cached =
        readCachedAnswer(cache, Opts, Label, plan.wallBudget);
    if (!cached)
      return cached.takeError();
    if (*cached)
      return std::move(**cached);
  }

  HerbieRunResult Run;
  ShardFiles files(Cores, plan.shards);
  if (!files.created()) {
    Run.Failure = "its input files or output directories could not be created";
    return Run;
  }
  announceRun(Opts, Label, Cores.size(), plan, command, keyDigest);
  ProcessOutcome ran = runShards(files, command, Opts.Program, plan.wallBudget);
  if (ran.executionFailed) {
    Run.Failure = ran.executionError;
    return Run;
  }
  Expected<std::optional<HerbieRunResult>> stopped =
      stoppedRun(ran, cache, Opts, Label, plan.wallBudget);
  if (!stopped)
    return stopped.takeError();
  if (*stopped)
    return std::move(**stopped);

  std::optional<json::Value> answer = readShardResults(files, Run.ResultsJson);
  if (!answer) {
    Run.Failure = noResultsReason(Opts, ran.status);
    return Run;
  }
  if (ran.status != 0)
    Run.Warning =
        ("Herbie " + Label + " exited with status " + Twine(ran.status) +
         " but wrote a usable results.json, which is used")
            .str();
  if (Opts.Print)
    llvm::errs() << "[poseidon] Herbie output: " << Run.ResultsJson << '\n';
  if (!cache.answerPath.empty()) {
    if (Error E = writeAnswer(Opts.CacheDir, cache.answerPath, cache.answeredBy,
                              std::move(*answer)))
      return std::move(E);
    if (Opts.Print)
      llvm::errs() << "[poseidon] Saved Herbie output to cache file "
                   << cache.answerPath << '\n';
  }
  if (Opts.Print)
    llvm::errs() << "[poseidon] Herbie " << Label << ": completed in "
                 << ran.seconds
                 << " s, a fresh answer of the Herbie with digest "
                 << binaryDigest << '\n';
  Run.Result = HerbieRunResult::Outcome::Ok;
  Run.CacheFile = cache.answerPath;
  return Run;
}

} // namespace poseidon
