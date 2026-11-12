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
    if (Name.contains("6Kokkos7Threads21impl_thread_pool_sizeE") ||
        Name.contains("6Kokkos7Threads11concurrencyE") ||
        (Name.contains("6Kokkos4Impl15ThreadsInternal") &&
         (Name.contains("5startE") || Name.contains("14resize_scratchE") ||
          Name.contains("19root_reduce_scratchE") ||
          Name.contains("10get_threadE") || Name.contains("5fenceE") ||
          Name.contains("14internal_fenceE"))) ||
        (Name.contains("6Kokkos7Threads") &&
         (Name.contains("17impl_static_fenceE") || Name.contains("5fenceE")) &&
         !Name.contains("19profile_fence_event")))
      return true;
  }
  return false;
}

/// Kokkos::Threads internal fence calls (`ThreadsInternal::fence` and
/// `ThreadsInternal::internal_fence`), handled in tandem with
/// `ThreadsInternal::start`.
inline bool isKokkosThreadsFenceCall(llvm::StringRef Name,
                                     llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos4Impl15ThreadsInternal5fenceEv" ||
      Name == "_ZN6Kokkos4Impl15ThreadsInternal14internal_fenceEv")
    return true;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::Impl::ThreadsInternal::fence(") ||
         startsWith(K, "Kokkos::Impl::ThreadsInternal::internal_fence(");
}

inline bool isKokkosFenceCall(llvm::StringRef Name, llvm::StringRef NormDem) {
  if (isKokkosThreadsFenceCall(Name, NormDem))
    return false;
  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::fence(") ||
         startsWith(K, "Kokkos::Threads::fence(") ||
         startsWith(K, "Kokkos::Threads::impl_static_fence(");
}

inline bool isKokkosThreadsPoolSizeQuery(llvm::StringRef Name,
                                         llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos7Threads21impl_thread_pool_sizeEi" ||
      Name == "_ZNK6Kokkos7Threads11concurrencyEv")
    return true;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::Threads::impl_thread_pool_size(") ||
         startsWith(K, "Kokkos::Threads::concurrency(");
}

/// Team-level barriers (`HostThreadTeamMember::team_barrier`,
/// `HostThreadTeamData::team_rendezvous[_release]`). Replayed verbatim in the
/// reverse pass so reverse team loops keep the same synchronization.
inline bool isKokkosTeamBarrierCall(llvm::StringRef Name,
                                    llvm::StringRef NormDem) {
  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (K.empty())
    return false;
  if ((startsWith(K, "Kokkos::Impl::HostThreadTeamMember<") ||
       startsWith(K, "Kokkos::Impl::ThreadsExecTeamMember::") ||
       startsWith(K, "Kokkos::Impl::OpenMPExecTeamMember::")) &&
      (K.contains("::team_barrier(") || K.contains("::team_rendezvous(") ||
       K.contains("::team_rendezvous_release(")))
    return true;
  if (startsWith(K, "Kokkos::Impl::HostThreadTeamData::") &&
      (K.contains("::team_rendezvous(") ||
       K.contains("::team_rendezvous_release(") ||
       K.contains("::rendezvous(") || K.contains("::rendezvous_release(")))
    return true;
  return false;
}

inline bool isKokkosInactiveCall(llvm::StringRef Name,
                                 llvm::StringRef NormDem) {
  if (startsWith(Name, "__enzyme_kokkos_") ||
      isKokkosFenceCall(Name, NormDem) ||
      isKokkosThreadsFenceCall(Name, NormDem) ||
      isKokkosTeamBarrierCall(Name, NormDem) ||
      isKokkosThreadsPoolSizeQuery(Name, NormDem))
    return true;
  if (NormDem.empty())
    return false;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  if (K.empty())
    return false;
  static const char *const InactivePrefixes[] = {
      "Kokkos::Threads::Threads(",
      "Kokkos::Threads::~Threads(",
      "Kokkos::Threads::name(",
      "Kokkos::Threads::print_configuration(",
      "Kokkos::Threads::impl_thread_pool_rank",
      "Kokkos::Threads::impl_hardware_thread_id(",
      "Kokkos::Threads::impl_max_hardware_threads(",
      "Kokkos::Threads::impl_instance_id(",
      "Kokkos::Impl::ThreadsInternal::verify_is_process(",
      "Kokkos::Impl::ThreadsInternal::is_process(",
      "Kokkos::Impl::ThreadsInternal::is_initialized(",
      "Kokkos::Impl::ThreadsInternal::print_configuration(",
      "Kokkos::Impl::ThreadsInternal::global_lock(",
      "Kokkos::Impl::ThreadsInternal::global_unlock(",
  };
  for (const char *P : InactivePrefixes) {
    if (startsWith(K, P))
      return true;
  }
  return false;
}

inline bool isKokkosThreadsStart(llvm::StringRef Name,
                                 llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos4Impl15ThreadsInternal5startEPFvRS1_PKvES4_")
    return true;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::Impl::ThreadsInternal::start(");
}

inline bool isKokkosThreadsResizeScratch(llvm::StringRef Name,
                                         llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos4Impl15ThreadsInternal14resize_scratchEmm")
    return true;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::Impl::ThreadsInternal::resize_scratch(");
}

inline bool isKokkosThreadsRootReduceScratch(llvm::StringRef Name,
                                             llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos4Impl15ThreadsInternal19root_reduce_scratchEv")
    return true;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::Impl::ThreadsInternal::root_reduce_scratch(");
}

inline bool isKokkosThreadsGetThread(llvm::StringRef Name,
                                     llvm::StringRef NormDem) {
  if (Name == "_ZN6Kokkos4Impl15ThreadsInternal10get_threadEi")
    return true;
  llvm::StringRef K = stripReturnTypeToNamespace(NormDem, "Kokkos::");
  return startsWith(K, "Kokkos::Impl::ThreadsInternal::get_thread(");
}

} // namespace enzyme_kokkos

#endif // ENZYME_KOKKOS_UTILS_H
