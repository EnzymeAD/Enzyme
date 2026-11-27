//===- HerbieRun.h - Running Herbie and reading its answers ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs the Herbie binary on a set of FPCore cores and returns its
// results.json, through the result cache: one file per question, holding the
// answer and the digest of the Herbie that gave it. The caller passes the
// options that say how to run Herbie. Reads what Herbie answered: the tests of
// its results.json, and each expression of an answer as an expression graph
// over the values of the site.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_ALGEBRAIC_HERBIE_RUN_H
#define POSEIDON_TECHNIQUES_ALGEBRAIC_HERBIE_RUN_H

#include "Core/FPCore.h"
#include "Core/FPNode.h"
#include "Core/Technique.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
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
  static inline char ID;
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

/// A point of a core's cost-accuracy curve: [cost, error in bits].
struct HerbiePoint {
  double Cost = 0, ErrorBits = 0;
};

/// An alternative Herbie lists for a core: [cost, error in bits, expression].
/// An entry without the expression is no candidate.
struct HerbieAlternative {
  HerbiePoint Point;
  std::optional<std::string> Expr;
};

/// A core's cost-accuracy list: [initial, best, alternatives].
struct HerbieCostAccuracy {
  HerbiePoint Initial, Best;
  std::vector<HerbieAlternative> Alternatives;
};

/// What Herbie found for a core it has an expression for.
struct HerbieRewrites {
  std::string BestExpr;
  /// The bits of the core's precision, which the errors are out of.
  double Bits = 0;
  HerbieCostAccuracy CostAccuracy;
};

/// One test of a Herbie results file: the answer for the core it names.
struct HerbieCoreAnswer {
  std::string Name;
  /// The core's index, which is its name.
  size_t Core = 0;
  /// What Herbie reports about its search, printed when present.
  std::optional<std::string> Status;
  std::optional<double> Time, Start, End;
  /// None when Herbie's output is "#f".
  std::optional<HerbieRewrites> Rewrites;
};

/// Reads the Herbie results file `Contents`, the answers for `NumCores` cores
/// named by their index.
llvm::Expected<std::vector<HerbieCoreAnswer>>
readHerbieResults(llvm::StringRef Contents, size_t NumCores);

/// The expression graph of the FPCore expression `expr`, over the values its
/// symbols name in `G`, which owns the nodes the parse creates. An error,
/// saying what and where, for an expression that is not one the evaluators and
/// the lowering model: an unknown symbol or operator, an operator of an
/// unmodelled format or with the wrong operand count, a truth value where a
/// number belongs or the reverse, or a malformed literal, approx form or array.
/// Each `#s(approx spec result)` adds its (spec, result) pair to `approxSpecs`
/// when given; with `subgraphInputs`, an operation that recomputes the
/// definition of one of those inputs is that input's node.
llvm::Expected<FPNode *> parseHerbieExpr(
    llvm::StringRef expr, FPGraph &G,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr,
    const llvm::SetVector<llvm::Value *> *subgraphInputs = nullptr);

/// The FPCore expression `expr` as C, one string per element of an array and
/// one otherwise; empty when `expr` names a symbol `G` does not know or does
/// not parse. The parse adds to `G`, as parseHerbieExpr does.
llvm::SmallVector<std::string, 1> fpcoreToC(
    llvm::StringRef expr, FPGraph &G, const CRendering &how,
    llvm::StringRef resultDtype = "",
    llvm::SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs =
        nullptr);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_ALGEBRAIC_HERBIE_RUN_H
