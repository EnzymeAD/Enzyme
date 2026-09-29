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
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_FLANG_RUNTIME_H
#define ENZYME_FLANG_RUNTIME_H

#include "llvm/ADT/StringRef.h"

#include <array>

/// A flang runtime function whose derivative replays it on shadows.
struct FlangShadowReplay {
  const char *name;
  /// Arguments that are replaced by their shadows (descriptors, or the data
  /// address of PointerAssociateScalar); -1 marks an unused slot. All other
  /// arguments (bounds, type info, stat, errmsg, source location, molds) are
  /// passed unchanged.
  std::array<int, 2> shadowArgs;
  /// Whether the call allocates the data of argument 0: the new shadow memory
  /// is then zeroed, as the tangent of freshly allocated memory is zero.
  bool allocates;
};

static const FlangShadowReplay FlangShadowReplays[] = {
    // allocatable.h
    {"_FortranAAllocatableInitIntrinsic", {0, -1}, false},
    {"_FortranAAllocatableInitCharacter", {0, -1}, false},
    {"_FortranAAllocatableInitDerived", {0, -1}, false},
    {"_FortranAAllocatableInitIntrinsicForAllocate", {0, -1}, false},
    {"_FortranAAllocatableInitCharacterForAllocate", {0, -1}, false},
    {"_FortranAAllocatableInitDerivedForAllocate", {0, -1}, false},
    {"_FortranAAllocatableApplyMold", {0, -1}, false},
    {"_FortranAAllocatableSetBounds", {0, -1}, false},
    {"_FortranAAllocatableSetDerivedLength", {0, -1}, false},
    {"_FortranAAllocatableAllocate", {0, -1}, true},
    {"_FortranAAllocatableAllocateSource", {0, 1}, false},
    {"_FortranAMoveAlloc", {0, 1}, false},
    {"_FortranAAllocatableDeallocate", {0, -1}, false},
    {"_FortranAAllocatableDeallocatePolymorphic", {0, -1}, false},
    {"_FortranAAllocatableDeallocateNoFinal", {0, -1}, false},
    // pointer.h
    {"_FortranAPointerNullifyIntrinsic", {0, -1}, false},
    {"_FortranAPointerNullifyCharacter", {0, -1}, false},
    {"_FortranAPointerNullifyDerived", {0, -1}, false},
    {"_FortranAPointerSetBounds", {0, -1}, false},
    {"_FortranAPointerSetDerivedLength", {0, -1}, false},
    {"_FortranAPointerApplyMold", {0, -1}, false},
    {"_FortranAPointerAssociateScalar", {0, 1}, false},
    {"_FortranAPointerAssociate", {0, 1}, false},
    {"_FortranAPointerAssociateLowerBounds", {0, 1}, false},
    {"_FortranAPointerAssociateRemapping", {0, 1}, false},
    {"_FortranAPointerAssociateRemappingMonomorphic", {0, 1}, false},
    {"_FortranAPointerAllocate", {0, -1}, true},
    {"_FortranAPointerAllocateSource", {0, 1}, false},
    {"_FortranAPointerDeallocate", {0, -1}, false},
    {"_FortranAPointerDeallocatePolymorphic", {0, -1}, false},
    // derived-api.h
    {"_FortranAInitialize", {0, -1}, false},
    {"_FortranAInitializeClone", {0, 1}, false},
    {"_FortranADestroy", {0, -1}, false},
    {"_FortranADestroyWithoutFinalization", {0, -1}, false},
    // assign.h (_FortranAAssign itself is handled on its own)
    {"_FortranAAssignTemporary", {0, 1}, false},
    {"_FortranACopyInAssign", {0, 1}, false},
    {"_FortranACopyOutAssignDirect", {0, 1}, false},
    // transformational.h
    {"_FortranAShallowCopyDirect", {0, 1}, false},
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
