//=- HerbieRun.cpp - Running Herbie, with its result cache -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HerbieRun.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <optional>

using namespace llvm;

namespace poseidon {

std::string readFileDigest(StringRef Path) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    return std::string();
  return llvm::utohexstr(llvm::xxh3_64bits((*Buf)->getBuffer()));
}

// A Herbie cache entry is stamped with the digest of the binary that produced
// it: version strings do not change between upstream commits, and one
// binary's rewrites must not be replayed as another's.
std::string herbieBinaryDigest(StringRef Program) {
  static std::map<std::string, std::string> Digests;
  auto It = Digests.find(Program.str());
  if (It == Digests.end())
    It = Digests.emplace(Program.str(), readFileDigest(Program)).first;
  return It->second;
}

SmallVector<std::string> herbieSearchArgs(const HerbieRunOptions &Opts) {
  // Herbie prunes alternatives by tree cost unless told otherwise, paying a
  // shared subterm once per occurrence; the emitted code computes it once.
  return {"--seed",       std::to_string(Opts.Seed),
          "--num-points", std::to_string(Opts.NumPoints),
          "--num-iters",  std::to_string(Opts.NumIters),
          "--num-enodes", std::to_string(Opts.NumEnodes),
          "--disable",    "generate:proofs",
          "--enable",     "reduce:dag-cost"};
}

void addHerbieKeyParts(CacheKey &Key, const HerbieRunOptions &Opts) {
  Key.add("Herbie binary", herbieBinaryDigest(Opts.Program));
  Key.add("Herbie platform", Opts.Platform.Name + " " + Opts.Platform.Digest);
}

HerbieRunResult runHerbie(const std::vector<std::string> &inputExprs,
                          const HerbieRunOptions &Opts, StringRef Label) {
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
                          inputExprs.size()));
  const unsigned threadsPerShard = std::max<unsigned>(
      1, std::min<size_t>((numThreads + shards - 1) / shards,
                          (inputExprs.size() + shards - 1) / shards));
  unsigned coreTimeout = (unsigned)Opts.CoreTimeout;
  unsigned wallBudget = 0; // 0 = wait forever
  if (subgraphBudget > 0) {
    coreTimeout =
        std::min<unsigned>((unsigned)Opts.CoreTimeout, subgraphBudget);
    unsigned perShard = (unsigned)((inputExprs.size() + shards - 1) / shards);
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
                   << "'"
                   << (Platform.Path.empty()
                           ? std::string()
                           : " from " + Platform.Path + " (digest " +
                                 Platform.Digest + ")")
                   << "\n";
  }

  HerbieRunResult Run;
  std::string &content = Run.ResultsJson;
  std::string cacheFilePath;

  // A result is filed under everything that decides it: the cores, the
  // binary, its search arguments and the platform it ranks candidates by.
  // The platform name leads the file name so that one device's results are
  // told apart from another's at a glance.
  std::string joinedCores;
  for (const auto &expr : inputExprs) {
    joinedCores += expr;
    joinedCores += '\n';
  }
  CacheKey key;
  addHerbieKeyParts(key, Opts);
  key.add("Herbie arguments", join(herbieSearchArgs(Opts), " "));
  key.addContent("cores", joinedCores);
  const std::string keyDigest = key.digest();
  std::string timeoutMarkerPath;

  if (!Opts.CacheDir.empty()) {
    std::string keyStem = Platform.Name + "_" + keyDigest;
    cacheFilePath = Opts.CacheDir + "/cachedHerbieOutput_" + keyStem + ".txt";
    // A timeout gets its own marker so it never looks like an empty result.
    timeoutMarkerPath =
        Opts.CacheDir + "/cachedHerbieTimeout_" + keyStem + ".txt";
    std::ifstream cacheFile(cacheFilePath);
    if (cacheFile) {
      content.assign((std::istreambuf_iterator<char>(cacheFile)),
                     std::istreambuf_iterator<char>());
      cacheFile.close();
      llvm::errs() << "Using cached Herbie output from " << cacheFilePath
                   << "\n";
      llvm::errs() << "Herbie output: " << content << "\n";
      Run.Result = Outcome::Ok;
      return Run;
    }
    // A recorded timeout is a result too: without it every re-solve pays the
    // full wall-clock budget again to learn the same thing.
    std::ifstream markerFile(timeoutMarkerPath);
    unsigned spentBudget = 0;
    if (markerFile >> spentBudget && wallBudget > 0 &&
        spentBudget >= wallBudget) {
      llvm::errs() << "[poseidon] Herbie " << Label << ": cached TIMEOUT ("
                   << spentBudget
                   << " s budget already spent on this exact input, "
                   << timeoutMarkerPath
                   << "). No algebraic candidates for this subgraph. "
                      "Raise -poseidon-herbie-subgraph-timeout above "
                   << spentBudget << " to retry it.\n";
      Run.Result = Outcome::Timeout;
      return Run;
    }
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
              << EC.message() << "\n";
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
    for (size_t k = s; k < inputExprs.size(); k += shards)
      input << inputExprs[k] << "\n";
    input.close();
    if (!input)
      setupFailed = true;
  }
  if (setupFailed) {
    llvm::errs() << "Failed to create the Herbie input files or output "
                    "directories.\n";
    removeShardFiles();
    return Run;
  }

  if (Opts.Print) {
    llvm::errs() << "Executing Herbie (" << shards
                 << " process(es), one input file and output directory "
                    "each) with arguments: ";
    for (const auto &arg : BaseArgs)
      llvm::errs() << arg << " ";
    llvm::errs() << "\n";
  }
  llvm::errs() << "[poseidon] Herbie " << Label << ": " << inputExprs.size()
               << " FPCore(s) in " << shards << " process(es) of "
               << threadsPerShard << " worker(s), per-core timeout "
               << coreTimeout << " s, wall budget "
               << (wallBudget ? std::to_string(wallBudget) + " s"
                              : std::string("unbounded"))
               << ", key " << keyDigest << "\n";

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
  int RC = 0;
  for (auto &PI : procs) {
    std::optional<unsigned> waitFor;
    if (ExecutionFailed || killed) {
      waitFor = 1;
    } else if (wallBudget) {
      double spent = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - herbieStart)
                         .count();
      waitFor = spent + 1 >= wallBudget ? 1u : (unsigned)(wallBudget - spent);
    }
    std::string WaitErr;
    llvm::sys::ProcessInfo R = llvm::sys::Wait(PI, waitFor, &WaitErr);
    if (R.ReturnCode == -2)
      killed = true;
    else if (R.ReturnCode != 0 && RC == 0)
      RC = R.ReturnCode;
  }
  double herbieSecs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - herbieStart)
                          .count();

  if (ExecutionFailed) {
    llvm::errs() << "Execution failed: " << ErrMsg << "\n";
    removeShardFiles();
    return Run;
  }
  if (killed) {
    llvm::errs() << "[poseidon] Herbie " << Label << ": KILLED after "
                 << herbieSecs << " s (wall budget " << wallBudget
                 << " s). No algebraic candidates for this subgraph; the "
                    "other candidate families are unaffected. Raise "
                    "-poseidon-herbie-subgraph-timeout to spend more, or set "
                    "it to 0 for the historical unbounded behaviour.\n";
    removeShardFiles();
    if (!timeoutMarkerPath.empty()) {
      if (auto EC = llvm::sys::fs::create_directories(Opts.CacheDir, true))
        llvm::errs() << "Warning: Could not create cache directory: "
                     << EC.message() << "\n";
      std::ofstream markerFile(timeoutMarkerPath);
      if (markerFile) {
        markerFile << wallBudget << "\n";
        markerFile.close();
        llvm::errs() << "Recorded the timeout in " << timeoutMarkerPath << "\n";
      }
    }
    Run.Result = Outcome::Timeout;
    return Run;
  }

  // One process: its results.json verbatim. Several: their tests merged into
  // the first one's report.
  std::optional<json::Object> merged;
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
      std::string where = Platform.Name;
      if (!Platform.Path.empty())
        where += ", " + Platform.Path;
      report_fatal_error(
          Twine("Poseidon: Herbie exited with status ") + Twine(RC) +
          " and produced no usable results.json for " + Label + " (platform " +
          where + "). Its own diagnostics are on stderr above.");
    }
    if (shards == 1) {
      content = (*Buf)->getBuffer().str();
      break;
    }
    if (!merged) {
      merged = std::move(*parsed->getAsObject());
      continue;
    }
    json::Array *dst = merged->getArray("tests");
    for (auto &t : *parsed->getAsObject()->getArray("tests"))
      dst->push_back(std::move(t));
  }
  if (merged) {
    content.clear();
    raw_string_ostream os(content);
    os << json::Value(std::move(*merged));
  }
  removeShardFiles();

  llvm::errs() << "Herbie output: " << content << "\n";

  if (!Opts.CacheDir.empty()) {
    if (auto EC = llvm::sys::fs::create_directories(Opts.CacheDir, true))
      llvm::errs() << "Warning: Could not create cache directory: "
                   << EC.message() << "\n";
    std::ofstream cacheFile(cacheFilePath);
    if (!cacheFile) {
      llvm_unreachable("Failed to open cache file for writing");
    } else {
      cacheFile << content;
      cacheFile.close();
      llvm::errs() << "Saved Herbie output to cache file " << cacheFilePath
                   << "\n";
    }
    // A stale timeout marker under this key is left in place: the read path
    // consults it only when the output file is absent.
  }
  llvm::errs() << "[poseidon] Herbie " << Label << ": completed in "
               << herbieSecs << " s\n";

  Run.Result = Outcome::Ok;
  return Run;
}

} // namespace poseidon
