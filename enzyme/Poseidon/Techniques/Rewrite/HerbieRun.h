//===- HerbieRun.h - Running Herbie, with its result cache ----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_TECHNIQUES_REWRITE_HERBIE_RUN_H
#define POSEIDON_TECHNIQUES_REWRITE_HERBIE_RUN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace poseidon {

struct Options;

/// The Herbie platform a search runs against: the cost table Herbie ranks its
/// rewrite candidates by. It is part of the cache key: a rewrite Herbie
/// proposed against another device's costs is that device's answer.
struct HerbiePlatform {
  /// The platform's name, part of the cache file name; "default" is Herbie's
  /// own platform, for a host cost model.
  std::string Name;
  /// The .rkt handed to `--platform`; empty means pass Name instead.
  std::string Path;
  /// Content digest of Path, so a recalibrated device (same Name, different
  /// prices) is a different key. Empty when Path is empty.
  std::string Digest;
};

/// The digest of a file's contents, which is never empty; "" when the file
/// cannot be read.
std::string readFileDigest(llvm::StringRef Path);

/// The digests of the Herbie binaries a compile ran, by path.
using HerbieBinaryDigests = std::map<std::string, std::string>;

/// Why a build configured without Herbie has none to run, and how to give it
/// one.
inline constexpr llvm::StringLiteral kBuiltWithoutHerbie =
    "this Poseidon was built without Herbie (configure it with "
    "-DPOSEIDON_HERBIE_BINARY=<herbie> or -DPOSEIDON_BUILD_HERBIE=ON)";

/// What a compile keeps between its Herbie runs: the platform Herbie searches
/// against, derived from the cost model once it is resolved, and the digest of
/// each Herbie binary run.
struct HerbieRunState {
  /// The platform the compile's searches run against, once resolved.
  HerbiePlatform Platform;
  /// Whether `Platform` has been derived from the cost model.
  bool PlatformResolved = false;
  /// The digest of each Herbie binary run, by path.
  HerbieBinaryDigests HerbieDigests;
  /// The answers searches took, by origin, since they were last reported.
  unsigned Cached = 0, Fresh = 0;
  /// Whether the compile has said it proposes no rewrites for want of a Herbie.
  bool LackReported = false;
};

/// How one Herbie run went and what it answered.
struct HerbieRunResult {
  /// How the run ended.
  enum class Outcome {
    /// `ResultsJson` is Herbie's results.json, run now or read from the cache.
    Ok,
    /// Herbie ran out of the wall-clock budget and was stopped. Nothing is
    /// cached.
    Timeout,
    /// Herbie ended by a signal the runner did not send; `Crash` names it.
    /// Nothing is cached.
    Crashed,
    /// Herbie could not be started, or it exited without a usable
    /// results.json; `Failure` says which.
    Failed,
  };
  /// How this run ended.
  Outcome Result = Outcome::Failed;
  /// Where an Ok result came from.
  enum class Origin { Cache, Fresh };
  /// Where this Ok result came from.
  Origin From = Origin::Fresh;
  /// Herbie's results.json, for an Ok result; empty otherwise.
  std::string ResultsJson;
  /// The cache file an Ok result was read from or saved to; empty when it is
  /// not cached.
  std::string CacheFile;
  /// The signal a Crashed run ended by; empty otherwise.
  std::string Crash;
  /// Why a Failed run failed; empty otherwise.
  std::string Failure;
  /// What the caller reports as a warning: why a timeout or a crash left the
  /// cores without an answer, that Herbie's per-core timeout left some of them
  /// without one, or that Herbie exited with a nonzero status after writing a
  /// usable answer. Empty when there is nothing to report.
  std::string Warning;
};

/// Runs Herbie on `Cores`, each a complete (FPCore ...), through the cache: the
/// answer this Herbie gave, else a run, whose answer replaces the cached one
/// unless a core in it timed out.
/// When the Herbie of `O` names no file, the cached answer is the one any
/// Herbie gave, and a question the cache does not answer is an error naming its
/// cache file. `TaylorTerms`, when given, is Herbie's --num-taylor-terms.
/// `Label` names the cores in diagnostics, e.g. "subgraph 0 of foo". A cache
/// file, or a temporary file or directory for Herbie, that cannot be written is
/// an error that names the path.
llvm::Expected<HerbieRunResult>
runHerbie(const std::vector<std::string> &Cores, const Options &O,
          const HerbiePlatform &Platform, std::optional<unsigned> TaylorTerms,
          llvm::StringRef Label, HerbieBinaryDigests &Digests);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_REWRITE_HERBIE_RUN_H
