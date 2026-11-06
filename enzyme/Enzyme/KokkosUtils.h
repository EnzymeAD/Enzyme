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
  if (startsWith(Name, "_ZNSt") && Name.contains("5mutex") &&
      (endsWith(Name, "4lockEv") || endsWith(Name, "6unlockEv") ||
       endsWith(Name, "8try_lockEv")))
    return true;
  if (Name.contains("5desul4Impl")) {
    if (Name.contains("22host_atomic_fetch_oper") ||
        Name.contains("24device_atomic_fetch_oper") ||
        Name.contains("lock_address") || Name.contains("unlock_address") ||
        Name.contains("init_lock_arrays") ||
        Name.contains("finalize_lock_arrays") || Name.contains("ensure_"))
      return true;
  }
  return false;
}

inline bool isKokkosFenceCall(llvm::StringRef Name, llvm::StringRef NormDem) {
  (void)Name;
  if (NormDem.empty())
    return false;
  llvm::StringRef D = stripReturnTypeToNamespace(NormDem, "desul::Impl::");
  return startsWith(D, "desul::Impl::host_atomic_thread_fence(") ||
         startsWith(D, "desul::Impl::device_atomic_thread_fence(");
}

inline bool isKokkosMutexCall(llvm::StringRef Name, llvm::StringRef NormDem) {
  if (Name == "pthread_mutex_lock" || Name == "pthread_mutex_unlock" ||
      Name == "pthread_mutex_trylock" || Name == "pthread_mutex_init" ||
      Name == "pthread_mutex_destroy")
    return true;
  if (!NormDem.empty()) {
    llvm::StringRef S = stripReturnTypeToNamespace(NormDem, "std::");
    if (startsWith(S, "std::mutex::") ||
        startsWith(S, "std::recursive_mutex::") ||
        startsWith(S, "std::lock_guard<std::mutex>::") ||
        startsWith(S, "std::unique_lock<std::mutex>::") ||
        startsWith(S, "std::scoped_lock<std::mutex>::"))
      return true;
  }
  return false;
}

inline bool isKokkosInactiveCall(llvm::StringRef Name,
                                 llvm::StringRef NormDem) {
  if (startsWith(Name, "__enzyme_kokkos_") ||
      isKokkosFenceCall(Name, NormDem) || isKokkosMutexCall(Name, NormDem))
    return true;
  if (NormDem.empty())
    return false;
  llvm::StringRef D = stripReturnTypeToNamespace(NormDem, "desul::Impl::");
  return startsWith(D, "desul::Impl::init_lock_arrays") ||
         startsWith(D, "desul::Impl::finalize_lock_arrays") ||
         startsWith(D, "desul::Impl::ensure_") ||
         startsWith(D, "desul::Impl::HostLocks::get_host_locks_") ||
         startsWith(D, "desul::Impl::HostLocks::get_host_lock_");
}

/// Identifies Desul address lock / unlock functions (`lock_address*` -> false,
/// `unlock_address*` -> true).
inline std::optional<bool> getDesulLockIsUnlock(llvm::StringRef Name,
                                                llvm::StringRef NormDem) {
  (void)Name;
  if (NormDem.empty())
    return std::nullopt;
  llvm::StringRef D = stripReturnTypeToNamespace(NormDem, "desul::Impl::");
  if (startsWith(D, "desul::Impl::lock_address"))
    return false;
  if (startsWith(D, "desul::Impl::unlock_address"))
    return true;
  return std::nullopt;
}

enum class DesulAtomicOpKind {
  None,
  Add,
  Sub,
  Other,
};

inline DesulAtomicOpKind classifyDesulAtomicFetchOper(llvm::StringRef NormDem) {
  if (NormDem.empty())
    return DesulAtomicOpKind::None;
  llvm::StringRef D = stripReturnTypeToNamespace(NormDem, "desul::Impl::");
  if (!startsWith(D, "desul::Impl::host_atomic_fetch_oper<") &&
      !startsWith(D, "desul::Impl::device_atomic_fetch_oper<"))
    return DesulAtomicOpKind::None;
  if (D.contains("_add_fetch_operator<"))
    return DesulAtomicOpKind::Add;
  if (D.contains("_sub_fetch_operator<"))
    return DesulAtomicOpKind::Sub;
  return DesulAtomicOpKind::Other;
}

enum class DesulPTXAtomicKind {
  None,
  Fence,
  Load,
  Add,
};

struct DesulPTXAtomicInfo {
  DesulPTXAtomicKind Kind = DesulPTXAtomicKind::None;
  unsigned BitWidth = 0;
};

/// Recognizes Desul CUDA PTX inline asm patterns (membar/fence, atomic ld,
/// and atom.add/red.add).
inline DesulPTXAtomicInfo classifyDesulPTXInlineAsm(const llvm::InlineAsm *IA) {
  if (!IA)
    return {};
  llvm::StringRef Asm = IA->getAsmString().empty()
                            ? llvm::StringRef()
                            : llvm::StringRef(IA->getAsmString()).trim();
  if (Asm.empty())
    return {};

  if (startsWith(Asm, "membar.") || startsWith(Asm, "fence."))
    return {DesulPTXAtomicKind::Fence, 0};

  auto detectWidth = [](llvm::StringRef S) -> unsigned {
    if (S.contains(".b64") || S.contains(".f64") || S.contains(".u64") ||
        S.contains(".s64"))
      return 64;
    if (S.contains(".b32") || S.contains(".f32") || S.contains(".u32") ||
        S.contains(".s32"))
      return 32;
    return 0;
  };

  if ((startsWith(Asm, "atom.") || startsWith(Asm, "red.")) &&
      Asm.contains(".add.") && (Asm.contains(".f32") || Asm.contains(".f64")))
    return {DesulPTXAtomicKind::Add, detectWidth(Asm)};
  if (startsWith(Asm, "ld.") && Asm.contains("[%1]")) {
    unsigned BW = detectWidth(Asm);
    if (BW == 32 || BW == 64)
      return {DesulPTXAtomicKind::Load, BW};
  }
  return {};
}

} // namespace enzyme_kokkos

#endif // ENZYME_KOKKOS_UTILS_H
