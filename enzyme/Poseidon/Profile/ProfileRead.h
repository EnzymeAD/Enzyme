//=- ProfileRead.h - Profiling utilities for Poseidon ----------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares profiling-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PROFILE_READ_H
#define POSEIDON_PROFILE_READ_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <limits>
#include <map>
#include <string>

namespace llvm {
class Function;
class Instruction;
class Module;
} // namespace llvm

namespace poseidon {

struct ProfileInfo {
  double minRes;
  double maxRes;
  double sumValue; // Sum of values (not abs)
  double sumSens;  // Sum of sensitivity scores = |grad * value|
  double sumGrad;  // Sum of gradients (not abs); the accuracy weight
  // Sum of |gradient|; negative means not recorded (older profiles) and
  // consumers fall back to |sumGrad|, which cancels for an adjoint that
  // alternates in sign.
  double sumAbsGrad;
  uint64_t exec; // 64-bit: a matmul site can exceed 2^32 execs, and a 32-bit
                 // field landing on a multiple of 2^32 truncates to 0.

  llvm::SmallVector<double, 2> minOperands;
  llvm::SmallVector<double, 2> maxOperands;
  llvm::SmallVector<double, 2>
      minMagOperands; // smallest nonzero |operand| (0=unknown)

  ProfileInfo()
      : minRes(std::numeric_limits<double>::max()),
        maxRes(std::numeric_limits<double>::lowest()), sumValue(0.0),
        sumSens(0.0), sumGrad(0.0), sumAbsGrad(-1.0), exec(0) {}
};

// A site's profile records by slot index, iterated in slot order: what is
// summed over them and reported about them does not depend on hashing.
using ProfileMap = std::map<size_t, ProfileInfo>;

// Launch geometry observed during profiling (max over launches).
struct FunctionProfileHeader {
  uint32_t maxBlockDim[3] = {0, 0, 0}; // [x, y, z]
  uint32_t maxGridDim[3] = {
      0, 0, 0}; // [x, y, z] launch grid (CTA count); 0 = unknown
  uint64_t launchCount = 0;
  // Static facts the profile-gen compile knew and the profile-use compile
  // cannot recompute; slot -> profile-scale reduction trip count.
  std::map<size_t, unsigned> redTrip;
  // Digest of the canonical form the slots were numbered against; empty in a
  // profile written before the field existed.
  std::string canonicalHash;
  // Site id the perturbation probe is armed with, -1 when the profile carries
  // no SiteId line.
  int siteId = -1;
  // Measured relative condition number of the declared quantity of interest
  // with respect to this site's outputs, written by the profiling run. 1.0
  // when the profile carries no Kappa line, which is the
  // no-cross-site-weighting case.
  double kappa = 1.0;
  // False when the profile carries no Kappa line: no declared quantity of
  // interest and so no cross-site weighting, or one left unmeasured.
  bool hasKappa = false;
  // "ok", "below-noise", "nonlinear" or "diverged" with a Kappa line;
  // kKappaUnmeasured without one when the run declared a quantity of interest
  // and did not measure the condition number; empty otherwise.
  std::string kappaMark;
};

inline constexpr llvm::StringLiteral kKappaUnmeasured = "unmeasured";

void parseProfileFile(const std::string &profilePath, ProfileMap &profileMap,
                      FunctionProfileHeader *header = nullptr);

// The static header of every instrumented site names the profile format in a
// line `ProfileFormat = <n>`. Profiles without it are format 1, written before
// the profiling runtime left a profile for a site it never executed; from
// format 2 it does.
inline constexpr llvm::StringLiteral kProfileFormatKey = "ProfileFormat";
inline constexpr unsigned kProfileFormat = 2;

// Whether `profilePath` is the profile the profiling runtime writes for a site
// it instrumented and never executed: the site's header, an `Executed = 0`
// line, and no records. Reads no further than the header. A profile that says
// both is an error at the site `F`.
bool isNeverExecutedProfile(const std::string &profilePath,
                            const llvm::Function &F);

// Whether a profile in `dir` names its format, so that a site the profiled run
// never executed would have left a file there. Reads headers only, once per
// directory: `known` holds the answers given.
bool profileDirNamesFormat(llvm::StringMap<bool> &known, llvm::StringRef dir);

// An error at the site `F` unless the profile's slot indices were numbered
// against the canonical form `canonicalHash` names.
void requireCanonicalHash(const FunctionProfileHeader &header,
                          llvm::StringRef canonicalHash,
                          llvm::StringRef profilePath, const llvm::Function &F);

// An error at the site `F` when the profiling run declared a quantity of
// interest and did not measure the site's condition number against it.
void requireMeasuredKappa(const FunctionProfileHeader &header,
                          llvm::StringRef profilePath, const llvm::Function &F);

// Degenerate-adjoint guard. A candidate is priced as (local error) * |sum of
// gradients|, so a value whose profiled gradient is ~0 is priced free; that
// happens both when the value genuinely has no influence and when the adjoint
// seed lies in the left null space of the site (an all-ones seed on any
// partition-of-unity operator with a derivative). The floor is on the
// sensitivity |gradient x value|: the gradients of values of very different
// magnitude are not comparable, their sensitivities are. Per function:
//   ref      = max over profiled instructions of sumSens / exec
//   floor(v) = 1e-6 * ref * exec_v / max |v|
// and an instruction whose summed |gradient| is below its floor is weighted at
// the floor, written back into sumGrad so every consumer sees it. A
// per-execution sensitivity below 1e-9 of ref is reported. A value that is
// always zero is left as it is. Returns the number of instructions whose
// weight was raised.
unsigned applySensitivityFloor(ProfileMap &profileMap,
                               llvm::StringRef functionName);

// Per-output accounting (-poseidon-output-profiles). Each directory holds this
// site's profile recorded with one output seeded alone. An output's modelled
// error is its first-order error over its own total sensitivity, so every
// instruction is weighted by the largest of its per-output gradients, each
// divided by that output's total; summed over a site this bounds the worst
// output's error, and the site's accuracy scale is 1.
void mergeOutputProfiles(ProfileMap &profileMap,
                         llvm::ArrayRef<std::string> dirs,
                         llvm::StringRef fileName,
                         llvm::StringRef canonicalHash,
                         const llvm::Function &F);

size_t readProfIdxMetadata(const llvm::Instruction *I);

// Non-aborting variant: false when I is null or lacks the metadata.
bool tryReadProfIdxMetadata(const llvm::Instruction *I, size_t &out);

// Profile-gen only: one poseidonProfileBlockDimsCUDA call per function carrying
// value probes, so launch geometry is recorded per function rather than in a
// module-global bucket.
void injectBlockDimProbes(llvm::Module &M);

// Profile-gen only, host module only: call the FP profiler's CUDA registration
// from the top of main (see the definition for why not a static constructor);
// whether it placed the call.
bool injectHostProfilerInit(llvm::Module &M);

// Profile-gen only, device module only: pre-declare the probes with
// enzyme_inactive so AD never differentiates them.
void predeclareInactiveProfilerProbes(llvm::Module &M);

} // namespace poseidon
#endif // POSEIDON_PROFILE_READ_H
