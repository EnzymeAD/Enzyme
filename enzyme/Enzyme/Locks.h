//===- Locks.h - Recognize lock / unlock primitives ------------*- C++ -*-===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A region locked in the forward pass must be locked again, in reverse order,
// in the reverse pass (in the manner of `__kmpc_critical` /
// `__kmpc_end_critical`): the reverse of an unlock is a (blocking) lock and
// the reverse of a lock that acquired the lock is an unlock. This header
// recognizes the lock and unlock entry points of the supported primitives and
// names their counterpart: std::mutex / std::recursive_mutex (libstdc++,
// libc++ and its std::__u ABI), pthread mutexes, and the Desul (Kokkos
// atomics) address locks.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_LOCKS_H
#define ENZYME_LOCKS_H

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

#include "Utils.h"

namespace locks {

/// How a lock call reports whether it acquired the lock.
enum class Acquires {
  Always, ///< blocking lock (void or ignored result)
  IfTrue, ///< `i1` result, true when acquired (try_lock, Desul lock_address)
  IfZero, ///< `int` result, 0 when acquired (pthread_mutex_trylock)
};

/// The result type of a lock or unlock entry point.
enum class Result { Void, I1, I32 };

struct LockCall {
  /// Whether this is the lock or the unlock entry point.
  bool IsLock;
  /// For a lock, how it reports acquisition.
  Acquires Acq;
  /// Whether the lock entry point is a single attempt that must be repeated
  /// to block (Desul lock_address), as opposed to a blocking lock.
  bool Spin;
  /// Whether the primitive itself provides no ordering and the surrounding
  /// code fences explicitly (Desul): the reverse then fences too.
  bool Fence;
  /// Mangled name of the counterpart (the blocking lock for an unlock, the
  /// unlock for a lock) and its result type.
  std::string Counterpart;
  Result CounterpartResult;
};

/// Classify `Name` as a lock / unlock entry point, if it is one.
inline std::optional<LockCall> classifyLockCall(llvm::StringRef Name) {
  // Desul address locks: lock_address<Scope>(void*, Scope) -> bool,
  // unlock_address<Scope>(void*, Scope) -> void. As function templates the
  // return type is mangled (`EEb` / `EEv`).
  if (startsWith(Name, "_ZN5desul4Impl12lock_addressI")) {
    std::string C = Name.str();
    C.replace(C.find("12lock_addressI"), 15, "14unlock_addressI");
    size_t P = C.find("EEbPv");
    if (P != std::string::npos)
      C.replace(P, 5, "EEvPv");
    return LockCall{true, Acquires::IfTrue, true, true, C, Result::Void};
  }
  if (startsWith(Name, "_ZN5desul4Impl14unlock_addressI")) {
    std::string C = Name.str();
    C.replace(C.find("14unlock_addressI"), 17, "12lock_addressI");
    size_t P = C.find("EEvPv");
    if (P != std::string::npos)
      C.replace(P, 5, "EEbPv");
    return LockCall{false, Acquires::Always, true, true, C, Result::I1};
  }

  // pthread mutexes.
  if (Name == "pthread_mutex_lock")
    return LockCall{true,  Acquires::Always,       false,
                    false, "pthread_mutex_unlock", Result::I32};
  if (Name == "pthread_mutex_trylock")
    return LockCall{true,  Acquires::IfZero,       false,
                    false, "pthread_mutex_unlock", Result::I32};
  if (Name == "pthread_mutex_unlock")
    return LockCall{false, Acquires::Always,     false,
                    false, "pthread_mutex_lock", Result::I32};

  // std::mutex / std::recursive_mutex methods: `_ZNSt` [`3__1` | `3__u`]
  // (`5mutex` | `15recursive_mutex`) (`4lock` | `6unlock` | `8try_lock`) `Ev`.
  llvm::StringRef Rest = Name;
  if (!Rest.consume_front("_ZNSt"))
    return std::nullopt;
  if (!Rest.consume_front("3__1"))
    Rest.consume_front("3__u");
  if (!Rest.consume_front("5mutex") && !Rest.consume_front("15recursive_mutex"))
    return std::nullopt;
  std::string Class = Name.drop_back(Rest.size()).str();
  if (Rest == "4lockEv")
    return LockCall{true,  Acquires::Always,    false,
                    false, Class + "6unlockEv", Result::Void};
  if (Rest == "8try_lockEv")
    return LockCall{true,  Acquires::IfTrue,    false,
                    false, Class + "6unlockEv", Result::Void};
  if (Rest == "6unlockEv")
    return LockCall{false, Acquires::Always,  false,
                    false, Class + "4lockEv", Result::Void};
  return std::nullopt;
}

} // namespace locks

#endif // ENZYME_LOCKS_H
