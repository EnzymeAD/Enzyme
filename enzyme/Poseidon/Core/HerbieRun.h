//===- HerbieRun.h - Running Herbie, with its result cache ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs the Herbie binary on a set of FPCore cores and returns its
// results.json, through the result cache: one file per question, holding the
// answer and the digest of the Herbie that gave it. Nothing here reads LLVM IR
// or Poseidon's command-line flags; the caller says how to run Herbie.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_HERBIERUN_H
#define POSEIDON_HERBIERUN_H

#include "CacheKey.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace poseidon {

// The Herbie platform a search runs against: the cost table Herbie ranks its
// rewrite candidates by. It is part of the cache key: a rewrite Herbie
// proposed against another device's costs is that device's answer.
struct HerbiePlatform {
  // The platform's name, part of the cache file name; "default" is Herbie's
  // own platform, for a host cost model.
  std::string Name;
  // The .rkt handed to `--platform`; empty means pass Name instead.
  std::string Path;
  // Content digest of Path, so a recalibrated device (same Name, different
  // prices) is a different key. Empty when Path is empty.
  std::string Digest;
};

// How to run Herbie.
struct HerbieRunOptions {
  std::string Program;
  HerbiePlatform Platform;

  // What decides the rewrites Herbie returns, and so the cache key.
  unsigned Seed = 0;
  int NumPoints = 0;
  int NumIters = 0;
  int NumEnodes = 0;
  // Herbie's --num-taylor-terms; none leaves Herbie its own cap.
  std::optional<unsigned> TaylorTerms;

  // What decides only whether and how soon Herbie returns.
  // Herbie's per-core --timeout, in seconds.
  int CoreTimeout = 0;
  // Worker threads in all, and the processes they are split across; 0
  // processes is one per thread.
  int NumThreads = 1;
  unsigned Processes = 0;
  // The wall-clock budget, in seconds, the per-core timeout is clamped to and
  // the processes are killed after; 0 waits for as long as Herbie takes.
  unsigned SubgraphTimeout = 0;

  // The directory of cached results; empty for none.
  std::string CacheDir;
  // Print the command line and the platform.
  bool Print = false;
};

// The digest of a file's contents, or "" when it cannot be read.
std::string readFileDigest(llvm::StringRef Path);

// The digests of the Herbie binaries a compile ran, by path.
using HerbieBinaryDigests = std::map<std::string, std::string>;

// What a compile keeps between its Herbie runs: the platform Herbie searches
// against, derived from the cost model once it is resolved, and the digest of
// each Herbie binary run.
struct HerbieRunState {
  HerbiePlatform Platform;
  bool PlatformResolved = false;
  HerbieBinaryDigests HerbieDigests;
  // The answers searches took, by origin, since they were last reported.
  unsigned Cached = 0, Fresh = 0;
};

// The parts of a cache key that say which Herbie searched: its binary and the
// platform it ranks candidates by.
void addHerbieKeyParts(CacheKey &Key, const HerbieRunOptions &Opts,
                       HerbieBinaryDigests &Digests);

struct HerbieRunResult {
  enum class Outcome {
    // `ResultsJson` is Herbie's results.json, run now or read from the cache.
    Ok,
    // Herbie ran out of the wall-clock budget, now or on an earlier run of
    // this exact input with the same budget.
    Timeout,
    // Herbie ended by a signal the runner did not send; `Crash` names it.
    // Nothing is cached.
    Crashed,
    // Herbie could not be started, its input files could not be written, or
    // it exited without a usable results.json; `Failure` says which.
    Failed,
  };
  Outcome Result = Outcome::Failed;
  // Where an Ok result came from.
  enum class Origin { Cache, Fresh };
  Origin From = Origin::Fresh;
  std::string ResultsJson;
  // The cache file an Ok result was read from or saved to; empty without a
  // cache.
  std::string CacheFile;
  std::string Crash;
  std::string Failure;
};

// Runs Herbie on `Cores`, each a complete (FPCore ...), through the cache: the
// answer this Herbie gave, else a run, whose answer replaces the cached one.
// When `Opts.Program` names no file, the cached answer is the one any Herbie
// gave, and a question the cache does not answer is an error naming its cache
// file. `Label` names the cores in diagnostics, e.g. "subgraph 0 of foo". A
// cache that cannot be written is an error that names the path.
llvm::Expected<HerbieRunResult> runHerbie(const std::vector<std::string> &Cores,
                                          const HerbieRunOptions &Opts,
                                          llvm::StringRef Label,
                                          HerbieBinaryDigests &Digests);

} // namespace poseidon
#endif // POSEIDON_HERBIERUN_H
