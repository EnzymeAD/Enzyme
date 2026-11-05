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
    if (startsWith(Name, "_ZN6Kokkos5fenceE") ||
        startsWith(Name, "_ZN6Kokkos6SerialC1Ev") ||
        startsWith(Name, "_ZN6Kokkos6SerialC2Ev") ||
        startsWith(Name, "_ZN6Kokkos6SerialD") ||
        Name.contains(
            "6Kokkos4Impl14SerialInternal23resize_thread_team_dataE") ||
        (Name.contains("6Kokkos6Serial") &&
         (Name.contains("17impl_static_fenceE") || Name.contains("5fenceE")) &&
         !Name.contains("19profile_fence_event")))
      return true;
  }
  return false;
}

inline bool isKokkosFenceCall(llvm::StringRef Name, llvm::StringRef NormDem) {
  (void)Name;
  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::fence(") ||
         startsWith(K, "Kokkos::memory_fence(") ||
         startsWith(K, "Kokkos::Serial::fence(") ||
         startsWith(K, "Kokkos::Serial::impl_static_fence(") ||
         startsWith(K, "Kokkos::Impl::SerialInternal::fence(");
}

inline bool isKokkosInactiveCall(llvm::StringRef Name,
                                 llvm::StringRef NormDem) {
  if (startsWith(Name, "__enzyme_kokkos_") || isKokkosFenceCall(Name, NormDem))
    return true;
  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (K.empty())
    return false;
  if (startsWith(K, "Kokkos::Serial::~Serial(") ||
      startsWith(K, "Kokkos::Serial::name(") ||
      startsWith(K, "Kokkos::Serial::concurrency(") ||
      startsWith(K, "Kokkos::Serial::print_configuration(") ||
      startsWith(K, "Kokkos::Serial::impl_thread_pool_size(") ||
      startsWith(K, "Kokkos::Serial::impl_thread_pool_rank(") ||
      startsWith(K, "Kokkos::Serial::impl_hardware_thread_id(") ||
      startsWith(K, "Kokkos::Serial::impl_max_hardware_threads(") ||
      startsWith(K, "Kokkos::Serial::impl_instance_id(") ||
      startsWith(
          K, "Kokkos::Impl::check_execution_space_constructor_precondition(") ||
      startsWith(
          K, "Kokkos::Impl::check_execution_space_destructor_precondition(") ||
      startsWith(K, "Kokkos::RangePolicy<") ||
      startsWith(K, "Kokkos::Impl::ImplRangePolicy<") ||
      startsWith(K, "Kokkos::MDRangePolicy<") ||
      startsWith(K, "Kokkos::TeamPolicy<") ||
      startsWith(K, "Kokkos::Impl::TeamPolicyInternal<"))
    return true;
  return false;
}

inline bool isKokkosSerialDefaultCtor(llvm::StringRef Name,
                                      llvm::StringRef NormDem) {
  return Name == "_ZN6Kokkos6SerialC1Ev" || Name == "_ZN6Kokkos6SerialC2Ev" ||
         Name == "__enzyme_kokkos_serial_init_d" ||
         NormDem == "Kokkos::Serial::Serial()";
}

inline bool isKokkosSerialInternalCtor(llvm::StringRef Name,
                                       llvm::StringRef NormDem) {
  return Name == "_ZN6Kokkos4Impl14SerialInternalC1Ev" ||
         Name == "_ZN6Kokkos4Impl14SerialInternalC2Ev" ||
         Name == "__enzyme_kokkos_serial_int_ctor_d" ||
         NormDem == "Kokkos::Impl::SerialInternal::SerialInternal()";
}

inline bool isKokkosSerialInternalDtor(llvm::StringRef Name,
                                       llvm::StringRef NormDem) {
  return Name == "_ZN6Kokkos4Impl14SerialInternalD1Ev" ||
         Name == "_ZN6Kokkos4Impl14SerialInternalD2Ev" ||
         Name == "__enzyme_kokkos_serial_int_dtor_d" ||
         NormDem == "Kokkos::Impl::SerialInternal::~SerialInternal()";
}

inline bool isKokkosSerialResizeTeamData(llvm::StringRef Name,
                                         llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos4Impl14SerialInternal23resize_thread_team_dataEmmmm" ||
      Name == "__enzyme_kokkos_serial_resize_d")
    return true;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K,
                    "Kokkos::Impl::SerialInternal::resize_thread_team_data(");
}

} // namespace enzyme_kokkos

#endif // ENZYME_KOKKOS_UTILS_H
