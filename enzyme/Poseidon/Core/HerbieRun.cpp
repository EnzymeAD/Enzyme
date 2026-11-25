//===- HerbieRun.cpp - Running Herbie, with its result cache --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HerbieRun.h"

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
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <optional>

using namespace llvm;

namespace poseidon {

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

void addHerbieKeyParts(CacheKey &Key, const HerbieRunOptions &Opts,
                       HerbieBinaryDigests &Digests) {
  Key.add("Herbie binary", herbieBinaryDigest(Opts.Program, Digests));
  Key.add("Herbie platform", Opts.Platform.Name + " " + Opts.Platform.Digest);
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

Expected<HerbieRunResult> runHerbie(const std::vector<std::string> &Cores,
                                    const HerbieRunOptions &Opts,
                                    StringRef Label,
                                    HerbieBinaryDigests &Digests) {
  using Outcome = HerbieRunResult::Outcome;
  const std::string &Program = Opts.Program;
  const HerbiePlatform &Platform = Opts.Platform;

  // Herbie runs these cores across Opts.NumThreads workers, so
  // ceil(cores/threads) rounds of the per-core timeout is what the invocation
  // can legitimately spend; the hard kill sits one margin above that.
  const unsigned subgraphBudget = Opts.SubgraphTimeout;
  const unsigned numThreads = std::max(1, Opts.NumThreads);
  const unsigned shards = (unsigned)std::max<size_t>(
      1, std::min<size_t>(Opts.Processes ? Opts.Processes : numThreads,
                          Cores.size()));
  const unsigned threadsPerShard = std::max<unsigned>(
      1, std::min<size_t>((numThreads + shards - 1) / shards,
                          (Cores.size() + shards - 1) / shards));
  unsigned coreTimeout = (unsigned)Opts.CoreTimeout;
  unsigned wallBudget = 0; // 0 = wait forever
  if (subgraphBudget > 0) {
    coreTimeout =
        std::min<unsigned>((unsigned)Opts.CoreTimeout, subgraphBudget);
    unsigned perShard = (unsigned)((Cores.size() + shards - 1) / shards);
    unsigned rounds = (perShard + threadsPerShard - 1) / threadsPerShard;
    if (rounds == 0)
      rounds = 1;
    wallBudget = coreTimeout * rounds + coreTimeout / 10 + 60;
    if (coreTimeout < (unsigned)Opts.CoreTimeout)
      llvm::errs() << "[poseidon] Herbie per-core --timeout clamped from "
                   << Opts.CoreTimeout << " s to " << coreTimeout
                   << " s by -poseidon-herbie-subgraph-timeout.\n";
  }

  SmallVector<std::string> BaseArgs = {
      Program,     "report",
      "--timeout", std::to_string(coreTimeout),
      "--threads", std::to_string(threadsPerShard)};
  BaseArgs.append(herbieSearchArgs(Opts));

  // "default" is Herbie's own choice of platform, which it makes when it is
  // not told one; a host cost model leaves it there.
  if (Platform.Name != "default" || !Platform.Path.empty()) {
    BaseArgs.push_back("--platform");
    BaseArgs.push_back(Platform.Path.empty() ? Platform.Name : Platform.Path);
    if (Opts.Print)
      llvm::errs() << "Poseidon: using Herbie platform '" << Platform.Name
                   << '\''
                   << (Platform.Path.empty()
                           ? std::string()
                           : " from " + Platform.Path + " (digest " +
                                 Platform.Digest + ")")
                   << '\n';
  }

  HerbieRunResult Run;
  std::string &content = Run.ResultsJson;
  std::string cacheFilePath;

  // An answer is filed under the question: the cores, the search arguments
  // and the platform Herbie ranks candidates by. The platform name leads the
  // file name so that one device's answers are told apart from another's at a
  // glance. The entry records the Herbie that gave it.
  std::string joinedCores;
  for (const auto &expr : Cores) {
    joinedCores += expr;
    joinedCores += '\n';
  }
  CacheKey question;
  question.add("Herbie platform", Platform.Name + " " + Platform.Digest);
  question.add("Herbie arguments", join(herbieSearchArgs(Opts), " "));
  question.addContent("cores", joinedCores);
  const std::string keyDigest = question.digest();
  // Without a Herbie to run, an answer any Herbie gave is the answer.
  const std::string binaryDigest = herbieBinaryDigest(Program, Digests);
  const bool haveHerbie = !binaryDigest.empty();
  CacheKey answeredBy;
  answeredBy.add("Herbie binary", binaryDigest);
  SmallVector<StringRef, 1> unchecked;
  if (!haveHerbie)
    unchecked.push_back("Herbie binary");
  std::string timeoutMarkerPath;

  if (!Opts.CacheDir.empty()) {
    std::string keyStem = Platform.Name + "_" + keyDigest;
    cacheFilePath = Opts.CacheDir + "/cachedHerbieOutput_" + keyStem + ".txt";
    // A timeout gets its own marker so it never looks like an empty result.
    timeoutMarkerPath =
        Opts.CacheDir + "/cachedHerbieTimeout_" + keyStem + ".txt";
    CacheEntryRead cached =
        readCacheEntry(cacheFilePath, answeredBy, unchecked);
    if (cached.value) {
      raw_string_ostream(content) << *cached.value;
      llvm::errs() << "Using cached Herbie output from " << cacheFilePath
                   << '\n';
      llvm::errs() << "Herbie output: " << content << '\n';
      Run.Result = Outcome::Ok;
      Run.From = HerbieRunResult::Origin::Cache;
      return Run;
    }
    if (!cached.unreadable.empty() && !haveHerbie)
      return createStringError(inconvertibleErrorCode(),
                               "the Herbie cache file " + cacheFilePath +
                                   " for " + Label + " is not a cache entry (" +
                                   cached.unreadable + ")");
    if (!cached.unreadable.empty() || !cached.differs.empty())
      llvm::errs() << "[poseidon] Herbie " << Label << ": the cache file "
                   << cacheFilePath << " holds "
                   << (cached.unreadable.empty()
                           ? "the answer of another Herbie"
                           : "no cache entry (" + cached.unreadable + ")")
                   << "; running Herbie to replace it\n";
  }

  if (!Opts.CacheDir.empty()) {
    // A recorded timeout is a result too: without it every re-solve pays the
    // full wall-clock budget again to learn the same thing.
    CacheEntryRead marker =
        readCacheEntry(timeoutMarkerPath, answeredBy, unchecked);
    std::optional<uint64_t> spentBudget =
        marker.value ? marker.value->getAsUINT64() : std::nullopt;
    if (spentBudget && wallBudget > 0 && *spentBudget >= wallBudget) {
      llvm::errs() << "[poseidon] Herbie " << Label << ": cached TIMEOUT ("
                   << *spentBudget
                   << " s budget already spent on this exact input, "
                   << timeoutMarkerPath
                   << "). No algebraic candidates for this subgraph. "
                      "Raise -poseidon-herbie-subgraph-timeout above "
                   << *spentBudget << " to retry it.\n";
      Run.Result = Outcome::Timeout;
      return Run;
    }
    if (!haveHerbie)
      return createStringError(
          inconvertibleErrorCode(),
          "no Herbie answer for " + Label + " is cached in " + cacheFilePath +
              ", and " +
              (Program.empty() ? "this Poseidon was built without Herbie"
                               : "there is no Herbie at " + Program) +
              " to compute it. Compute the answer on a machine with Herbie "
              "and copy its -poseidon-cache directory here, or turn "
              "algebraic rewriting off with -poseidon-enable-herbie=0.");
  }

  SmallVector<SmallString<64>> shardIn(shards), shardOut(shards);
  auto removeShardFiles = [&]() {
    for (unsigned s = 0; s < shards; ++s) {
      if (!shardIn[s].empty())
        llvm::sys::fs::remove(shardIn[s]);
      if (!shardOut[s].empty())
        if (auto EC = llvm::sys::fs::remove_directories(shardOut[s]))
          llvm::errs()
              << "Warning: Failed to remove temporary output directory: "
              << EC.message() << '\n';
    }
  };
  bool setupFailed = false;
  for (unsigned s = 0; s < shards && !setupFailed; ++s) {
    if (llvm::sys::fs::createUniqueFile("herbie_input_%%%%%%%%%%%%%%%%",
                                        shardIn[s],
                                        llvm::sys::fs::perms::owner_all) ||
        llvm::sys::fs::createUniqueDirectory("herbie_output_%%%%%%%%%%%%%%%%",
                                             shardOut[s])) {
      setupFailed = true;
      break;
    }
    std::ofstream input(shardIn[s].c_str());
    for (size_t k = s; k < Cores.size(); k += shards)
      input << Cores[k] << "\n";
    input.close();
    if (!input)
      setupFailed = true;
  }
  if (setupFailed) {
    Run.Failure = "its input files or output directories could not be created";
    removeShardFiles();
    return Run;
  }

  if (Opts.Print) {
    llvm::errs() << "Executing Herbie (" << shards
                 << " process(es), one input file and output directory "
                    "each) with arguments: ";
    for (const auto &arg : BaseArgs)
      llvm::errs() << arg << ' ';
    llvm::errs() << '\n';
  }
  llvm::errs() << "[poseidon] Herbie " << Label << ": " << Cores.size()
               << " FPCore(s) in " << shards << " process(es) of "
               << threadsPerShard << " worker(s), per-core timeout "
               << coreTimeout << " s, wall budget "
               << (wallBudget ? std::to_string(wallBudget) + " s"
                              : std::string("unbounded"))
               << ", key " << keyDigest << '\n';

  auto herbieStart = std::chrono::steady_clock::now();
  std::string ErrMsg;
  bool ExecutionFailed = false;
  SmallVector<llvm::sys::ProcessInfo> procs;
  for (unsigned s = 0; s < shards && !ExecutionFailed; ++s) {
    SmallVector<StringRef> Args;
    for (const auto &arg : BaseArgs)
      Args.emplace_back(arg);
    Args.push_back(shardIn[s]);
    Args.push_back(shardOut[s]);
    llvm::sys::ProcessInfo PI = llvm::sys::ExecuteNoWait(
        Program, Args, /*Env=*/std::nullopt,
        /*Redirects=*/{}, /*MemoryLimit=*/0, &ErrMsg, &ExecutionFailed);
    if (!ExecutionFailed)
      procs.push_back(PI);
  }
  bool killed = false;
  std::string crash;
  int RC = 0;
  for (auto &PI : procs) {
    std::optional<unsigned> waitFor;
    if (ExecutionFailed || killed || !crash.empty()) {
      waitFor = 1;
    } else if (wallBudget) {
      double spent = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - herbieStart)
                         .count();
      waitFor = spent + 1 >= wallBudget ? 1u : (unsigned)(wallBudget - spent);
    }
    std::string WaitErr;
    llvm::sys::ProcessInfo R = llvm::sys::Wait(PI, waitFor, &WaitErr);
    // Wait returns -2 both when its own alarm killed the process and when a
    // signal from elsewhere (the OOM killer's SIGKILL) ended it; only its
    // message tells the two apart.
    if (R.ReturnCode == -2) {
      if (StringRef(WaitErr).starts_with("Child timed out"))
        killed = true;
      else if (crash.empty())
        crash = "signal: " + WaitErr;
    } else if (R.ReturnCode != 0 && RC == 0) {
      RC = R.ReturnCode;
    }
  }
  double herbieSecs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - herbieStart)
                          .count();

  if (ExecutionFailed) {
    Run.Failure = ErrMsg;
    removeShardFiles();
    return Run;
  }
  // A crash is not a property of the input, so unlike a timeout it is not
  // recorded: the next compile runs Herbie again.
  if (!crash.empty()) {
    llvm::errs() << "[poseidon] Herbie " << Label << ": crashed (" << crash
                 << ") after " << herbieSecs
                 << " s. No algebraic candidates for this subgraph in this "
                    "compile; nothing is cached, so the next compile runs "
                    "Herbie again.\n";
    removeShardFiles();
    Run.Result = Outcome::Crashed;
    Run.Crash = crash;
    return Run;
  }
  if (killed) {
    llvm::errs() << "[poseidon] Herbie " << Label << ": KILLED after "
                 << herbieSecs << " s (wall budget " << wallBudget
                 << " s). No algebraic candidates for this subgraph; the "
                    "other candidate families are unaffected. Raise "
                    "-poseidon-herbie-subgraph-timeout to spend more, or set "
                    "it to 0 to wait for as long as Herbie takes.\n";
    removeShardFiles();
    if (!timeoutMarkerPath.empty()) {
      if (Error E = writeAnswer(Opts.CacheDir, timeoutMarkerPath, answeredBy,
                                (uint64_t)wallBudget))
        return std::move(E);
      llvm::errs() << "Recorded the timeout in " << timeoutMarkerPath << '\n';
    }
    Run.Result = Outcome::Timeout;
    return Run;
  }

  // One process: its results.json verbatim. Several: their tests merged into
  // the first one's report.
  std::optional<json::Value> answer;
  for (unsigned s = 0; s < shards; ++s) {
    auto Buf = MemoryBuffer::getFile(shardOut[s] + "/results.json");
    Expected<json::Value> parsed =
        Buf ? json::parse((*Buf)->getBuffer())
            : Expected<json::Value>(
                  createStringError(Buf.getError(), "no results.json"));
    if (!parsed || !parsed->getAsObject() ||
        !parsed->getAsObject()->getArray("tests")) {
      if (!parsed)
        consumeError(parsed.takeError());
      removeShardFiles();
      if (Opts.TaylorTerms && !takesTaylorTerms(Program)) {
        Run.Failure = "-poseidon-herbie-num-taylor-terms passes it "
                      "--num-taylor-terms, which it does not take ('report "
                      "--help' does not list it); use a Herbie built with the "
                      "option, or leave the flag out";
        return Run;
      }
      std::string where = Platform.Name;
      if (!Platform.Path.empty())
        where += ", " + Platform.Path;
      Run.Failure = (Twine("it exited with status ") + Twine(RC) +
                     " and wrote no usable results.json (platform " + where +
                     "); its own diagnostics are on stderr above")
                        .str();
      return Run;
    }
    if (!answer) {
      if (shards == 1)
        content = (*Buf)->getBuffer().str();
      answer = std::move(*parsed);
      continue;
    }
    json::Array *dst = answer->getAsObject()->getArray("tests");
    for (auto &t : *parsed->getAsObject()->getArray("tests"))
      dst->push_back(std::move(t));
  }
  if (shards > 1) {
    content.clear();
    raw_string_ostream(content) << *answer;
  }
  removeShardFiles();
  if (RC != 0)
    llvm::errs() << "[poseidon] Warning: Herbie " << Label
                 << " exited with status " << RC
                 << " but wrote a usable results.json; using it.\n";

  llvm::errs() << "Herbie output: " << content << '\n';

  if (!Opts.CacheDir.empty()) {
    if (Error E = writeAnswer(Opts.CacheDir, cacheFilePath, answeredBy,
                              std::move(*answer)))
      return std::move(E);
    llvm::errs() << "Saved Herbie output to cache file " << cacheFilePath
                 << '\n';
    // A stale timeout marker under this key is left in place: the read path
    // consults it only when the output file is absent.
  }
  llvm::errs() << "[poseidon] Herbie " << Label << ": completed in "
               << herbieSecs << " s, a fresh answer of the Herbie with digest "
               << binaryDigest << '\n';

  Run.Result = Outcome::Ok;
  return Run;
}

} // namespace poseidon
