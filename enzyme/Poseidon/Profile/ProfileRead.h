//===- ProfileRead.h - reading the profile of a site ----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The profile a profiling run writes for a site (runtime/fpprofiler), read and
// checked against the site's canonical form.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PROFILE_READ_H
#define POSEIDON_PROFILE_READ_H

#include "runtime/fpprofiler/FPProfileName.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <map>
#include <string>

namespace llvm {
class Function;
} // namespace llvm

namespace poseidon {

/// The record of one profiled instruction.
struct ProfileInfo {
  /// The smallest value the instruction produced over the profiled executions.
  double minRes = std::numeric_limits<double>::max();
  /// The largest value the instruction produced over the profiled executions.
  double maxRes = std::numeric_limits<double>::lowest();
  /// Sum of the sensitivities |gradient x value|.
  double sumSens = 0.0;
  /// Sum of the signed gradients of the value over its executions.
  double sumGrad = 0.0;
  /// Sum of |gradient|, the accuracy weight: the signed sum understates a value
  /// whose adjoint alternates in sign.
  double sumAbsGrad = 0.0;
  /// Number of times the instruction was executed in the profiling run.
  uint64_t exec = 0;
  /// Per operand; minMagOperands is the smallest nonzero |operand|, 0 if the
  /// operand was never nonzero.
  llvm::SmallVector<double, 2> minOperands, maxOperands, minMagOperands;

  /// The accuracy weight of the instruction, its summed |gradient|.
  double weight() const { return sumAbsGrad; }
};

/// A site's profile records by slot index, iterated in slot order: what is
/// summed over them and reported about them does not depend on hashing.
using ProfileMap = std::map<size_t, ProfileInfo>;

/// The KappaMark of a profile: how the profiling runs measured the condition
/// number of the declared quantity of interest.
enum class KappaMark { None, Ok, BelowNoise, Nonlinear, Diverged, Unmeasured };

/// The header of a site's profile.
struct FunctionProfileHeader {
  /// Launch geometry over the launches of a kernel (largest block and grid,
  /// number of launches), 0 where unknown; a device profile records it.
  uint32_t maxBlockDim[3] = {0, 0, 0};
  /// The largest grid a launch had, per dimension; 0 where unknown.
  uint32_t maxGridDim[3] = {0, 0, 0};
  /// Number of launches profiled; 0 where unknown.
  uint64_t launchCount = 0;
  /// Slot -> reduction trip count at profile scale, which the profile-use
  /// compile cannot recompute.
  std::map<size_t, unsigned> redTrip;
  /// Digest of the canonical form the slots were numbered against; empty when
  /// the profile carries none.
  std::string canonicalHash;
  /// Measured relative condition number of the declared quantity of interest
  /// with respect to this site's outputs; 1.0 when the profile carries no
  /// Kappa line.
  double kappa = 1.0;
  /// Whether the profile carries a Kappa line.
  bool hasKappa = false;
  /// How the run measured Kappa; Unmeasured without a Kappa line when the run
  /// declared a quantity of interest and did not measure the condition number;
  /// None when the profile carries no KappaMark line.
  KappaMark kappaMark = KappaMark::None;
  /// False for a site the profiling run instrumented and never executed.
  bool executed = true;
};

/// The word a KappaMark line spells KappaMark::Unmeasured with.
inline constexpr llvm::StringLiteral kKappaUnmeasured = kProfileKappaUnmeasured;

/// The word a profile's KappaMark line spells `M` with.
llvm::StringRef kappaMarkName(KappaMark M);

/// The records and the header of the profile at `profilePath`. A profile that
/// cannot be read, or does not parse, is an error at the site `F`.
llvm::Error parseProfileFile(const std::string &profilePath,
                             const llvm::Function &F, ProfileMap &profileMap,
                             FunctionProfileHeader &header);

/// An error at the site `F` unless the profile's slot indices were numbered
/// against the canonical form `canonicalHash` names.
llvm::Error requireCanonicalHash(const FunctionProfileHeader &header,
                                 llvm::StringRef canonicalHash,
                                 llvm::StringRef profilePath,
                                 const llvm::Function &F);

/// An error at the site `F` when the profiling run declared a quantity of
/// interest and did not measure the site's condition number against it.
llvm::Error requireMeasuredKappa(const FunctionProfileHeader &header,
                                 llvm::StringRef profilePath,
                                 const llvm::Function &F);

/// Floors the accuracy weight (summed |gradient|) of every instruction, so that
/// a value whose profiled gradient is ~0 is not priced free; that happens both
/// when the value has no influence and when the adjoint seed lies in the left
/// null space of the site (an all-ones seed on any partition-of-unity operator
/// with a derivative). The floor is on the sensitivity |gradient x value|, as
/// the gradients of values of very different magnitude are not comparable:
///   ref      = max over profiled instructions of sumSens / exec
///   floor(v) = 1e-6 * ref * exec_v / max |v|
/// A raised weight is written back into sumGrad and sumAbsGrad. A
/// per-execution sensitivity below 1e-9 of ref is reported. A value that is
/// always zero is left as it is. Returns the number of instructions whose
/// weight was raised.
unsigned applySensitivityFloor(ProfileMap &profileMap,
                               llvm::StringRef functionName, bool print);

/// Per-output accounting (-poseidon-output-profiles). Each directory holds this
/// site's profile recorded with one output seeded alone. An output's modelled
/// error is its first-order error over its own total sensitivity, so every
/// instruction is weighted by the largest of its per-output gradients, each
/// divided by that output's total; summed over a site this bounds the worst
/// output's error, and the site's accuracy scale is 1.
llvm::Error mergeOutputProfiles(ProfileMap &profileMap,
                                llvm::ArrayRef<std::string> dirs,
                                llvm::StringRef fileName,
                                llvm::StringRef canonicalHash,
                                const llvm::Function &F, bool print);

} // namespace poseidon
#endif // POSEIDON_PROFILE_READ_H
