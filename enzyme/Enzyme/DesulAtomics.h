//===- DesulAtomics.h - Recognize Desul (Kokkos) atomic symbols -*- C++ -*-===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kokkos implements its atomics on top of Desul. The relevant Desul entry
// points are function templates whose Itanium-mangled names start with
// `_ZN5desul4Impl` followed by the length-prefixed function name and the
// template arguments; they are matched here on the mangled name directly.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_DESUL_ATOMICS_H
#define ENZYME_DESUL_ATOMICS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AtomicOrdering.h"

#include "Utils.h"

namespace desul_atomics {

/// `desul::Impl::host_atomic_fetch_oper<Op, T, MemoryOrder, MemoryScope>` or
/// its `device_atomic_fetch_oper` counterpart.
inline bool isAtomicFetchOper(llvm::StringRef Name) {
  return startsWith(Name, "_ZN5desul4Impl22host_atomic_fetch_operI") ||
         startsWith(Name, "_ZN5desul4Impl24device_atomic_fetch_operI");
}

/// The linear `Op` of a Desul fetch_oper: `_add_fetch_operator` or
/// `_sub_fetch_operator`.
enum class LinearFetchOp { None, Add, Sub };

inline LinearFetchOp classifyLinearFetchOper(llvm::StringRef Name) {
  if (!isAtomicFetchOper(Name))
    return LinearFetchOp::None;
  if (Name.contains("19_add_fetch_operatorI"))
    return LinearFetchOp::Add;
  if (Name.contains("19_sub_fetch_operatorI"))
    return LinearFetchOp::Sub;
  return LinearFetchOp::None;
}

/// The `desul::MemoryOrder*` template argument of a Desul atomic symbol.
inline llvm::AtomicOrdering memoryOrderOf(llvm::StringRef Name) {
  if (Name.contains("18MemoryOrderAcquireE"))
    return llvm::AtomicOrdering::Acquire;
  if (Name.contains("18MemoryOrderReleaseE"))
    return llvm::AtomicOrdering::Release;
  if (Name.contains("17MemoryOrderAcqRelE"))
    return llvm::AtomicOrdering::AcquireRelease;
  if (Name.contains("17MemoryOrderSeqCstE"))
    return llvm::AtomicOrdering::SequentiallyConsistent;
  return llvm::AtomicOrdering::Monotonic;
}

} // namespace desul_atomics

#endif // ENZYME_DESUL_ATOMICS_H
