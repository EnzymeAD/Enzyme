//===- KokkosUtils.h - Helpers for Kokkos differentiation -------*- C++ -*-===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_KOKKOS_UTILS_H
#define ENZYME_KOKKOS_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"

#include <optional>
#include <string>

#include "Utils.h"

namespace enzyme_kokkos {

/// Quick pre-check before invoking llvm::demangle so non-Kokkos symbols pay
/// near-zero cost.
inline bool mayBeKokkosRelevantSymbol(llvm::StringRef Name) {
  return startsWith(Name, "__enzyme_kokkos_") || Name.contains("Kokkos") ||
         Name.contains("desul") || Name.contains("kokkosp") ||
         Name.contains("mutex") || Name.contains("cuda") ||
         Name.contains("hip");
}

/// Demangles \p Name (if potentially Kokkos/Desul/runtime related) using
/// \p DemangleFn and normalizes C++ ABI inline namespaces (`std::__u::`,
/// `std::__1::`, `std::__cxx11::` -> `std::`) and `> >` -> `>>`.
template <typename DemangleFnT>
inline std::string normalizeKokkosDemangledNameWith(llvm::StringRef Name,
                                                    DemangleFnT &&DemangleFn) {
  if (Name.empty() || !mayBeKokkosRelevantSymbol(Name))
    return {};
  std::string D = DemangleFn(Name);
  auto replaceAll = [&](const char *From, size_t FromLen, const char *To,
                        size_t ToLen) {
    size_t Pos = 0;
    while ((Pos = D.find(From, Pos)) != std::string::npos) {
      D.replace(Pos, FromLen, To, ToLen);
      Pos += ToLen;
    }
  };
  replaceAll("> >", 3, ">>", 2);
  replaceAll("std::__u::", 10, "std::", 5);
  replaceAll("std::__1::", 10, "std::", 5);
  replaceAll("std::__cxx11::", 14, "std::", 5);
  return D;
}

#define ENZYME_KOKKOS_DEMANGLE(Name)                                           \
  enzyme_kokkos::normalizeKokkosDemangledNameWith(                             \
      (Name), [](llvm::StringRef S) { return llvm::demangle(S.str()); })

/// Strip leading return type from a normalized demangled C++ function signature
/// up to the qualified name (e.g. `bool desul::Impl::lock_address<...>(...)` ->
/// `desul::Impl::lock_address<...>(...)`).
///
/// The return type may itself live in the namespace being searched for (e.g.
/// `Kokkos::Impl::SharedAllocationRecord<void, void>*
/// Kokkos::Impl::make_shared_allocation_record<...>(...)`), so rather than
/// searching for the first occurrence of \p NS we locate the qualified name
/// as the token following the last top-level space before the parameter list.
inline llvm::StringRef stripReturnTypeToNamespace(llvm::StringRef NormDem,
                                                  llvm::StringRef NS) {
  if (NormDem.empty())
    return {};
  size_t AngleDepth = 0;
  size_t ParenDepth = 0;
  size_t LastTopSpace = llvm::StringRef::npos;
  size_t PrevTopSpace = llvm::StringRef::npos;
  size_t i = 0;
  for (; i < NormDem.size(); ++i) {
    char c = NormDem[i];
    if (c == '<' && ParenDepth == 0) {
      AngleDepth++;
    } else if (c == '>' && ParenDepth == 0) {
      if (AngleDepth > 0)
        AngleDepth--;
    } else if (c == '(') {
      if (AngleDepth == 0 && ParenDepth == 0)
        break;
      ParenDepth++;
    } else if (c == ')') {
      if (ParenDepth > 0)
        ParenDepth--;
    } else if (c == ' ' && AngleDepth == 0 && ParenDepth == 0) {
      PrevTopSpace = LastTopSpace;
      LastTopSpace = i;
    }
  }
  // `operator new` / `operator delete` / `operator new[]` contain a space in
  // the function name itself.
  if (LastTopSpace != llvm::StringRef::npos &&
      endsWith(NormDem.substr(0, LastTopSpace), "operator"))
    LastTopSpace = PrevTopSpace;
  size_t Start = (LastTopSpace == llvm::StringRef::npos) ? 0 : LastTopSpace + 1;
  while (
      Start < NormDem.size() &&
      (NormDem[Start] == '*' || NormDem[Start] == '&' || NormDem[Start] == ' '))
    Start++;
  llvm::StringRef Res = NormDem.substr(Start);
  if (startsWith(Res, NS))
    return Res;
  // Fallback: the first occurrence of NS at a token boundary that is not
  // itself a return type (i.e. the demangled name has no top-level space).
  if (LastTopSpace == llvm::StringRef::npos && startsWith(NormDem, NS))
    return NormDem;
  return {};
}

inline bool isKokkosPreservedFunctionMangled(llvm::StringRef Name) {
  if (Name.empty() || !mayBeKokkosRelevantSymbol(Name))
    return false;
  if (Name.contains("6Kokkos")) {
    if (Name.contains("6Kokkos13kokkos_malloc") ||
        Name.contains("6Kokkos11kokkos_free") ||
        Name.contains("6Kokkos14kokkos_realloc") ||
        Name.contains("6Kokkos4Impl29make_shared_allocation_record") ||
        (Name.contains("6Kokkos4Impl22SharedAllocationRecordIv") &&
         !Name.contains("6Kokkos4Impl22SharedAllocationRecordIvvE4dataEv") &&
         !Name.contains("6Kokkos4Impl22SharedAllocationRecordIvvE4headEv") &&
         !Name.contains("6Kokkos4Impl22SharedAllocationRecordIvvE4sizeEv")) ||
        Name.contains("6Kokkos4Impl28SharedAllocationRecordCommon") ||
        Name.contains(
            "6Kokkos4Impl44HostInaccessibleSharedAllocationRecordCommon") ||
        Name.contains("16allocate_trackedE") ||
        Name.contains("18deallocate_trackedE") ||
        Name.contains("18reallocate_tracked"))
      return true;
  }
  return false;
}

inline bool isKokkosInactiveCall(llvm::StringRef Name,
                                 llvm::StringRef NormDem) {
  if (startsWith(Name, "__enzyme_kokkos_") ||
      Name == "_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9incrementEPS2_" ||
      Name == "_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9decrementEPS2_")
    return true;
  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (K.empty())
    return false;
  if (startsWith(
          K, "Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_") ||
      startsWith(
          K, "Kokkos::Impl::SharedAllocationRecord<void, void>::use_count(") ||
      startsWith(
          K, "Kokkos::Impl::SharedAllocationRecord<void, void>::is_sane(") ||
      startsWith(
          K, "Kokkos::Impl::SharedAllocationRecord<void, void>::increment(") ||
      startsWith(
          K, "Kokkos::Impl::SharedAllocationRecord<void, void>::decrement(") ||
      startsWith(K, "Kokkos::Impl::SharedAllocationRecord<void, "
                    "void>::print_host_accessible_records(") ||
      startsWith(K, "Kokkos::Impl::fill_host_accessible_header_info(") ||
      K.contains("::get_label(") || K.contains("::label(") ||
      K.contains("::use_count(") ||
      (K.contains("SharedAllocationRecord") && K.contains("::print_records(")))
    return true;
  return false;
}

enum class KokkosMemSpaceKind {
  Host,
  Cuda,
  CudaUVM,
  Hip,
};

inline KokkosMemSpaceKind inferKokkosMemSpace(llvm::StringRef NormDem) {
  if (NormDem.contains("CudaUVMSpace"))
    return KokkosMemSpaceKind::CudaUVM;
  if (NormDem.contains("CudaSpace"))
    return KokkosMemSpaceKind::Cuda;
  if (NormDem.contains("HIPSpace") || NormDem.contains("HIPManagedSpace"))
    return KokkosMemSpaceKind::Hip;
  return KokkosMemSpaceKind::Host;
}

enum class KokkosAllocKind {
  None,
  MakeSharedAllocRecord,
  SharedAllocRecordData,
  SharedAllocRecordIncrement,
  SharedAllocRecordDecrement,
  SharedAllocRecordBaseCtor,
  SharedAllocRecordCommonCtor,
  SharedAllocRecordDtor,
  SharedAllocRecordAllocate,
  SharedAllocRecordAllocateTracked,
  SharedAllocRecordDeallocateTracked,
  SharedAllocRecordReallocateTracked,
  SharedAllocRecordGetRecord,
  SpaceAllocate,
  SpaceDeallocate,
};

inline bool isKokkosFreshAllocationKind(KokkosAllocKind K) {
  return K == KokkosAllocKind::MakeSharedAllocRecord ||
         K == KokkosAllocKind::SharedAllocRecordAllocate ||
         K == KokkosAllocKind::SharedAllocRecordAllocateTracked ||
         K == KokkosAllocKind::SpaceAllocate;
}

inline KokkosAllocKind classifyKokkosAllocCall(llvm::StringRef Name,
                                               llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9incrementEPS2_")
    return KokkosAllocKind::SharedAllocRecordIncrement;
  if (Name == "_ZN6Kokkos4Impl22SharedAllocationRecordIvvE9decrementEPS2_")
    return KokkosAllocKind::SharedAllocRecordDecrement;
  if (Name == "_ZNK6Kokkos4Impl22SharedAllocationRecordIvvE4dataEv")
    return KokkosAllocKind::SharedAllocRecordData;

  if (NormDem.empty())
    return KokkosAllocKind::None;

  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (K.empty())
    return KokkosAllocKind::None;

  if (startsWith(K, "Kokkos::Impl::make_shared_allocation_record<"))
    return KokkosAllocKind::MakeSharedAllocRecord;
  if (startsWith(K, "Kokkos::Impl::SharedAllocationRecord<void, void>::data("))
    return KokkosAllocKind::SharedAllocRecordData;
  if (startsWith(
          K, "Kokkos::Impl::SharedAllocationRecord<void, void>::increment("))
    return KokkosAllocKind::SharedAllocRecordIncrement;
  if (startsWith(
          K, "Kokkos::Impl::SharedAllocationRecord<void, void>::decrement("))
    return KokkosAllocKind::SharedAllocRecordDecrement;
  if (startsWith(K, "Kokkos::Impl::SharedAllocationRecord<void, "
                    "void>::SharedAllocationRecord(") &&
      !startsWith(K, "Kokkos::Impl::SharedAllocationRecord<void, "
                     "void>::SharedAllocationRecord()"))
    return KokkosAllocKind::SharedAllocRecordBaseCtor;

  if (startsWith(K, "Kokkos::Impl::SharedAllocationRecordCommon<") ||
      startsWith(
          K, "Kokkos::Impl::HostInaccessibleSharedAllocationRecordCommon<") ||
      startsWith(K, "Kokkos::Impl::SharedAllocationRecord<")) {
    if (K.contains("::~SharedAllocationRecord"))
      return KokkosAllocKind::SharedAllocRecordDtor;
    if (K.contains("::SharedAllocationRecordCommon(") ||
        K.contains("::HostInaccessibleSharedAllocationRecordCommon("))
      return KokkosAllocKind::SharedAllocRecordCommonCtor;
    // Kokkos 4.x SharedAllocationRecord<Space,
    // void>::SharedAllocationRecord(...)
    if (!startsWith(K, "Kokkos::Impl::SharedAllocationRecord<void, void>") &&
        K.contains("::SharedAllocationRecord("))
      return KokkosAllocKind::SharedAllocRecordCommonCtor;
    if (K.contains("::allocate_tracked("))
      return KokkosAllocKind::SharedAllocRecordAllocateTracked;
    if (K.contains("::deallocate_tracked("))
      return KokkosAllocKind::SharedAllocRecordDeallocateTracked;
    if (K.contains("::reallocate_tracked"))
      return KokkosAllocKind::SharedAllocRecordReallocateTracked;
    if (K.contains("::allocate("))
      return KokkosAllocKind::SharedAllocRecordAllocate;
    if (K.contains("::get_record(") || K.contains("::find("))
      return KokkosAllocKind::SharedAllocRecordGetRecord;
  }

  if (startsWith(K, "Kokkos::kokkos_malloc"))
    return KokkosAllocKind::SharedAllocRecordAllocateTracked;
  if (startsWith(K, "Kokkos::kokkos_free"))
    return KokkosAllocKind::SharedAllocRecordDeallocateTracked;
  if (startsWith(K, "Kokkos::kokkos_realloc"))
    return KokkosAllocKind::SharedAllocRecordReallocateTracked;

  if (startsWith(K, "Kokkos::HostSpace::") ||
      startsWith(K, "Kokkos::CudaSpace::") ||
      startsWith(K, "Kokkos::CudaUVMSpace::") ||
      startsWith(K, "Kokkos::CudaHostPinnedSpace::") ||
      startsWith(K, "Kokkos::HIPSpace::") ||
      startsWith(K, "Kokkos::HIPHostPinnedSpace::") ||
      startsWith(K, "Kokkos::HIPManagedSpace::") ||
      startsWith(K, "Kokkos::Experimental::SYCLDeviceUSMSpace::") ||
      startsWith(K, "Kokkos::Experimental::SYCLSharedUSMSpace::") ||
      startsWith(K, "Kokkos::Experimental::SYCLHostUSMSpace::")) {
    if (K.contains("::allocate(") || K.contains("::impl_allocate("))
      return KokkosAllocKind::SpaceAllocate;
    if (K.contains("::deallocate(") || K.contains("::impl_deallocate("))
      return KokkosAllocKind::SpaceDeallocate;
  }

  return KokkosAllocKind::None;
}

} // namespace enzyme_kokkos

#endif // ENZYME_KOKKOS_UTILS_H
