//===- FlangRuntime.h - LLVM flang runtime functions known to Enzyme  -----===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// Functions of the LLVM flang runtime (flang-rt) that act on Fortran
// descriptors: allocation, pointer association, initialization and copies.
// Their forward-mode derivative is the same call on the shadow descriptors.
// In reverse mode the augmented pass replays them the same way; only the
// allocation group (structure changes, allocation, deallocation) is
// supported there, and the reverse pass undoes the shadow allocations.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_FLANG_RUNTIME_H
#define ENZYME_FLANG_RUNTIME_H

#include "llvm/ADT/StringRef.h"

#include <array>

/// What a replayed flang runtime function does to its first descriptor.
enum class FlangReplayKind {
  /// Changes the descriptor only (bounds, type, association, component
  /// initialization): replayed in the augmented pass, nothing in reverse.
  Structure,
  /// Allocates the data of argument 0: the new shadow memory is zeroed; the
  /// reverse pass deallocates it again.
  Allocate,
  /// Deallocates argument 0: the shadow memory is detached rather than freed,
  /// since the reverse pass still accumulates into it, and reattached in the
  /// reverse pass.
  Deallocate,
  /// Copies values, moves allocations or destroys components: no reverse
  /// rule yet, forward mode only.
  ForwardOnly,
};

/// A flang runtime function whose derivative replays it on shadows.
struct FlangShadowReplay {
  const char *name;
  /// Arguments that are replaced by their shadows (descriptors, or the data
  /// address of PointerAssociateScalar); -1 marks an unused slot. All other
  /// arguments (bounds, type info, stat, errmsg, source location, molds) are
  /// passed unchanged.
  std::array<int, 2> shadowArgs;
  FlangReplayKind kind;
};

static const FlangShadowReplay FlangShadowReplays[] = {
    // allocatable.h
    {"_FortranAAllocatableInitIntrinsic", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAAllocatableInitCharacter", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAAllocatableInitDerived", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAAllocatableInitIntrinsicForAllocate",
     {0, -1},
     FlangReplayKind::Structure},
    {"_FortranAAllocatableInitCharacterForAllocate",
     {0, -1},
     FlangReplayKind::Structure},
    {"_FortranAAllocatableInitDerivedForAllocate",
     {0, -1},
     FlangReplayKind::Structure},
    {"_FortranAAllocatableApplyMold", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAAllocatableSetBounds", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAAllocatableSetDerivedLength",
     {0, -1},
     FlangReplayKind::Structure},
    {"_FortranAAllocatableAllocate", {0, -1}, FlangReplayKind::Allocate},
    {"_FortranAAllocatableAllocateSource",
     {0, 1},
     FlangReplayKind::ForwardOnly},
    {"_FortranAMoveAlloc", {0, 1}, FlangReplayKind::ForwardOnly},
    {"_FortranAAllocatableDeallocate", {0, -1}, FlangReplayKind::Deallocate},
    {"_FortranAAllocatableDeallocatePolymorphic",
     {0, -1},
     FlangReplayKind::Deallocate},
    {"_FortranAAllocatableDeallocateNoFinal",
     {0, -1},
     FlangReplayKind::Deallocate},
    // pointer.h
    {"_FortranAPointerNullifyIntrinsic", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAPointerNullifyCharacter", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAPointerNullifyDerived", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAPointerSetBounds", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAPointerSetDerivedLength", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAPointerApplyMold", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAPointerAssociateScalar", {0, 1}, FlangReplayKind::Structure},
    {"_FortranAPointerAssociate", {0, 1}, FlangReplayKind::Structure},
    {"_FortranAPointerAssociateLowerBounds",
     {0, 1},
     FlangReplayKind::Structure},
    {"_FortranAPointerAssociateRemapping", {0, 1}, FlangReplayKind::Structure},
    {"_FortranAPointerAssociateRemappingMonomorphic",
     {0, 1},
     FlangReplayKind::Structure},
    {"_FortranAPointerAllocate", {0, -1}, FlangReplayKind::Allocate},
    {"_FortranAPointerAllocateSource", {0, 1}, FlangReplayKind::ForwardOnly},
    {"_FortranAPointerDeallocate", {0, -1}, FlangReplayKind::Deallocate},
    {"_FortranAPointerDeallocatePolymorphic",
     {0, -1},
     FlangReplayKind::Deallocate},
    // derived-api.h
    {"_FortranAInitialize", {0, -1}, FlangReplayKind::Structure},
    {"_FortranAInitializeClone", {0, 1}, FlangReplayKind::ForwardOnly},
    {"_FortranADestroy", {0, -1}, FlangReplayKind::ForwardOnly},
    {"_FortranADestroyWithoutFinalization",
     {0, -1},
     FlangReplayKind::ForwardOnly},
    // assign.h (_FortranAAssign itself is handled on its own)
    {"_FortranAAssignTemporary", {0, 1}, FlangReplayKind::ForwardOnly},
    {"_FortranACopyInAssign", {0, 1}, FlangReplayKind::ForwardOnly},
    {"_FortranACopyOutAssignDirect", {0, 1}, FlangReplayKind::ForwardOnly},
    // transformational.h
    {"_FortranAShallowCopyDirect", {0, 1}, FlangReplayKind::ForwardOnly},
};

/// The replay rule of a flang runtime function, or null.
static inline const FlangShadowReplay *
getFlangShadowReplay(llvm::StringRef name) {
  if (name.substr(0, 9) != "_FortranA")
    return nullptr;
  for (auto &R : FlangShadowReplays)
    if (name == R.name)
      return &R;
  return nullptr;
}

#endif // ENZYME_FLANG_RUNTIME_H
