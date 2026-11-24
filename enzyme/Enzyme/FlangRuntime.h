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
// Also helpers for the reverse-mode derivatives of assignments and SUM.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_FLANG_RUNTIME_H
#define ENZYME_FLANG_RUNTIME_H

#include "llvm/ADT/StringRef.h"

#include <array>

namespace llvm {
class Function;
class Module;
class Type;
} // namespace llvm

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

// Helpers for the reverse-mode derivatives of flang runtime functions. They
// are emitted into \p M on first use. Descriptors are passed as i8 pointers.

/// i8 *__enzyme_flang_descriptor_save(i8 *desc) returns a malloc'ed copy of
/// a descriptor (without addendum).
llvm::Function *getOrInsertFlangDescriptorSave(llvm::Module &M);

/// i8 *__enzyme_flang_shadow_save(i8 *shadow, i8 *primal) returns a
/// malloc'ed copy of the descriptor \p primal with the data address of
/// \p shadow. The other fields of a shadow descriptor need not be valid.
llvm::Function *getOrInsertFlangShadowSave(llvm::Module &M);

/// i8 *__enzyme_flang_shadow_realloc(i8 *shadow, i8 *to, i8 *before,
///                                   i8 *file, i32 line)
/// runs after an assignment to `to`, given a saved copy \p before of `to`
/// from before it. If the assignment (re)allocated `to`, or the shadow is
/// unallocated, it gives the shadow new, zeroed memory of the shape of `to`
/// and returns a saved copy of the old shadow descriptor, whose data it
/// leaves allocated. Returns null otherwise.
llvm::Function *getOrInsertFlangShadowAssignRealloc(llvm::Module &M);

/// void __enzyme_flang_assign_reverse(i8 *shadow, i8 *dto, i8 *dfrom,
///                                    i8 *old, i8 *file, i32 line)
/// is the reverse pass of `to = from`: it adds the adjoint of the elements
/// of dto to those of dfrom (if not null) and zeroes it. dto and dfrom are
/// from __enzyme_flang_shadow_save, and it frees them. If \p old is not null,
/// it then frees the data of dto and restores \p shadow from \p old.
llvm::Function *getOrInsertFlangAssignReverse(llvm::Module &M);

/// i8 *__enzyme_flang_mask_save(i8 *mask, i8 *array) returns a malloc'ed
/// byte per element of \p array, whether \p mask (a LOGICAL scalar or array
/// conforming with \p array) is true there; null if \p mask is null.
llvm::Function *getOrInsertFlangMaskSave(llvm::Module &M);

/// void __enzyme_flang_sum_reverse_<FPTy>(i8 *darray, i8 *bits, FPTy dret)
/// adds \p dret to each element of \p darray (from
/// __enzyme_flang_shadow_save)
/// where \p bits (from __enzyme_flang_mask_save, or null for all elements)
/// is set, and frees \p darray.
llvm::Function *getOrInsertFlangSumReverse(llvm::Module &M, llvm::Type *FPTy);

#endif // ENZYME_FLANG_RUNTIME_H
