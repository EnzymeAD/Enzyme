//===- EnzymeCallMarkers.h - The __enzyme_* call grammar --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The arguments of an `__enzyme_autodiff`/`__enzyme_fwddiff` call are a small
// language: marker globals that either name the activity of the argument after
// them or configure the call as a whole, the arguments they speak for, and the
// shadows that go with those.
//
// More than one place has to read it -- the LLVM pass that lowers the call, and
// the MLIR raising that turns it into an enzyme.autodiff/enzyme.fwddiff op --
// and a marker read as an argument, or an argument read as a marker, is a
// silently wrong derivative rather than an error. So the names, what each one
// takes, and where the shadows sit are written here once, over no IR in
// particular, and each reader drives its own walk with them.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_CALL_MARKERS_H
#define ENZYME_CALL_MARKERS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace enzyme_markers {

/// What a marker does to the call it appears in.
enum class MarkerRole {
  /// Names the activity of the argument that follows it.
  Activity,
  /// Says the shadows come after all of the primals rather than beside each
  /// one, in the same order.
  Interleave,
  /// Configures the call rather than an argument: nothing follows it that is
  /// an argument.
  CallFlag,
  /// Changes how the argument that follows is read without naming an activity.
  Modifier,
};

} // namespace enzyme_markers

/// Potential differentiable argument classifications
enum class DIFFE_TYPE {
  OUT_DIFF = 0, // add differential to an output struct. Only for scalar values
                // in ReverseMode variants.
  DUP_ARG = 1,  // duplicate the argument and store differential inside.
               // For references, pointers, or integers in ReverseMode variants.
               // For all types in ForwardMode variants.
  CONSTANT = 2,  // no differential. Usable everywhere.
  DUP_NONEED = 3 // duplicate this argument and store differential inside, but
                 // don't need the forward. Same as DUP_ARG otherwise.
};

namespace enzyme_markers {

struct MarkerInfo {
  MarkerRole role;
  /// Set where the role is Activity, and for the few call flags that also fix
  /// the activity of what they name.
  std::optional<DIFFE_TYPE> activity;
  /// Operands the marker itself takes, before whatever follows it.
  unsigned extraOperands;
};

/// The one description of the `__enzyme_*` argument markers.
///
/// Returns nothing for a name this does not know -- callers report that as the
/// error it is rather than carrying on, since guessing at an unknown marker is
/// how an argument gets read as a marker.
inline std::optional<MarkerInfo> lookupEnzymeMarker(llvm::StringRef name) {
  auto activity = [](DIFFE_TYPE a, unsigned extra = 0) {
    return MarkerInfo{MarkerRole::Activity, a, extra};
  };
  auto flag = [](unsigned extra = 0) {
    return MarkerInfo{MarkerRole::CallFlag, std::nullopt, extra};
  };
  auto flagWithActivity = [](DIFFE_TYPE a, unsigned extra) {
    return MarkerInfo{MarkerRole::CallFlag, a, extra};
  };

  return llvm::StringSwitch<std::optional<MarkerInfo>>(name)
      // Activities.
      .Case("enzyme_const", activity(DIFFE_TYPE::CONSTANT))
      .Case("enzyme_dup", activity(DIFFE_TYPE::DUP_ARG))
      .Case("enzyme_dupnoneed", activity(DIFFE_TYPE::DUP_NONEED))
      .Case("enzyme_out", activity(DIFFE_TYPE::OUT_DIFF))
      // Vector activities, each followed by the offset between lanes.
      .Case("enzyme_dupv", activity(DIFFE_TYPE::DUP_ARG, 1))
      .Case("enzyme_dupnoneedv", activity(DIFFE_TYPE::DUP_NONEED, 1))
      // Where the shadows are.
      .Case("enzyme_interleave",
            MarkerInfo{MarkerRole::Interleave, std::nullopt, 0})
      // How the argument after is read.
      .Case("enzyme_not_overwritten",
            MarkerInfo{MarkerRole::Modifier, std::nullopt, 0})
      // Whole-call flags that take nothing.
      .Case("enzyme_noret", flag())
      .Case("enzyme_nofree", flag())
      .Case("enzyme_runtime_activity", flag())
      .Case("enzyme_strong_zero", flag())
      .Case("enzyme_primal_return", flag())
      .Case("enzyme_const_return", flag())
      .Case("enzyme_active_return", flag())
      .Case("enzyme_dup_return", flag())
      // Whole-call flags that take a value.
      .Case("enzyme_byref", flag(1))
      .Case("enzyme_allocated", flag(1))
      .Case("enzyme_tape", flag(1))
      .Case("enzyme_width", flag(1))
      .Case("enzyme_interface", flag(1))
      .Case("enzyme_active_rand_var", flag(1))
      .Case("enzyme_trace", flagWithActivity(DIFFE_TYPE::CONSTANT, 1))
      .Case("enzyme_duptrace", flagWithActivity(DIFFE_TYPE::CONSTANT, 1))
      .Case("enzyme_likelihood", flagWithActivity(DIFFE_TYPE::CONSTANT, 1))
      .Case("enzyme_observations", flagWithActivity(DIFFE_TYPE::CONSTANT, 1))
      .Case("enzyme_duplikelihood", flagWithActivity(DIFFE_TYPE::DUP_ARG, 2))
      .Default(std::nullopt);
}

/// Whether an activity is one that a shadow goes with.
inline bool markerActivityTakesShadow(DIFFE_TYPE a) {
  return a == DIFFE_TYPE::DUP_ARG || a == DIFFE_TYPE::DUP_NONEED;
}

/// Where the primals stop and the shadows start.
struct InterleaveSplit {
  /// One past the last operand that can be a primal.
  unsigned primalEnd;
  /// The first shadow operand, meaningful only when `interleaved`.
  unsigned shadowStart;
  bool interleaved;
};

/// Find `enzyme_interleave` among the operands, which has to be done before
/// walking them: it says where the shadows are, and the walk needs that from
/// the first argument on.
///
/// `markerAt(i)` gives the marker name operand `i` reads, if it reads one.
template <typename MarkerAtFn>
InterleaveSplit findEnzymeInterleave(unsigned first, unsigned count,
                                     MarkerAtFn markerAt) {
  for (unsigned i = first; i < count; ++i) {
    std::optional<llvm::StringRef> name = markerAt(i);
    if (name && *name == "enzyme_interleave")
      return {/*primalEnd=*/i, /*shadowStart=*/i + 1, /*interleaved=*/true};
  }
  return {/*primalEnd=*/count, /*shadowStart=*/0, /*interleaved=*/false};
}

} // namespace enzyme_markers

#endif // ENZYME_CALL_MARKERS_H
