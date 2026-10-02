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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include <limits>
#include <map>
#include <string>
#include <unordered_map>

namespace llvm {
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
  // with respect to this site's outputs, written by
  // Poseidon/scripts/poseidon_probe.py. 1.0 when the profile carries no Kappa
  // line, which is the no-cross-site-weighting case.
  double kappa = 1.0;
  // False when the profile carries no Kappa line, which is the case with no
  // declared quantity of interest and so no cross-site weighting.
  bool hasKappa = false;
  // "ok", "below-noise", "nonlinear" or "diverged"; empty when there is no
  // Kappa line.
  std::string kappaMark;
};

void parseProfileFile(const std::string &profilePath,
                      std::unordered_map<size_t, ProfileInfo> &profileMap,
                      FunctionProfileHeader *header = nullptr);

// Degenerate-adjoint guard. A candidate is priced as (local error) * |sum of
// gradients|, so a value whose profiled gradient is ~0 is priced free; that
// happens both when the value genuinely has no influence and when the adjoint
// seed lies in the left null space of the site (an all-ones seed on any
// partition-of-unity operator with a derivative). Within one site every value
// reaches the output through a bounded chain, so per function:
//   ref  = max over profiled instructions of |sumGrad| / exec
//   w(v) = max(|sumGrad_v|, flags::GradFloorRatio * ref * exec_v)
// The floor is written back into sumGrad so every consumer sees it; a ratio
// below flags::GradNullRatio is reported (or refused under
// -poseidon-grad-floor-abort). Returns the number of instructions whose weight
// was raised.
unsigned applyGradientFloor(std::unordered_map<size_t, ProfileInfo> &profileMap,
                            llvm::StringRef functionName);

size_t readProfIdxMetadata(const llvm::Instruction *I);

// Non-aborting variant: false when I is null or lacks the metadata.
bool tryReadProfIdxMetadata(const llvm::Instruction *I, size_t &out);

// Profile-gen only: one poseidonProfileBlockDimsCUDA call per function carrying
// value probes, so launch geometry is recorded per function rather than in a
// module-global bucket.
void injectBlockDimProbes(llvm::Module &M);

// Profile-gen only, host module only: call the FP profiler's CUDA registration
// from the top of main (see the definition for why not a static constructor).
void injectHostProfilerInit(llvm::Module &M);

// Profile-gen only, device module only: pre-declare the probes with
// enzyme_inactive so AD never differentiates them.
void predeclareInactiveProfilerProbes(llvm::Module &M);

} // namespace poseidon
#endif // POSEIDON_PROFILE_READ_H
