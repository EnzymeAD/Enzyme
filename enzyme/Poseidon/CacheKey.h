//=- CacheKey.h - what a cached result was computed from ------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Both of Poseidon's caches, Herbie's results and the per-site DP tables, are
// keyed on the inputs the cached result was computed from, as named parts. A
// Herbie result is filed under the digest of its key. A DP table entry stores
// its key's parts, so a mismatch can say which input changed.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CACHEKEY_H
#define POSEIDON_CACHEKEY_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>
#include <utility>

namespace poseidon {

class CacheKey {
public:
  // Bumped whenever what a cached result means changes; every result written
  // under another format is a miss.
  static constexpr unsigned kFormat = 1;

  CacheKey();
  void add(llvm::StringRef name, llvm::StringRef value);
  // For a part too large to store, such as a file: kept as its digest.
  void addContent(llvm::StringRef name, llvm::StringRef bytes);

  std::string digest() const;
  llvm::json::Object toJSON() const;
  // The name of the first part `stored` disagrees on, "format" when it was
  // written under another format or under none; nullopt when it agrees.
  std::optional<std::string>
  firstDifference(const llvm::json::Object *stored) const;

private:
  llvm::SmallVector<std::pair<std::string, std::string>, 64> parts;
};

std::string contentDigest(llvm::StringRef bytes);

} // namespace poseidon
#endif // POSEIDON_CACHEKEY_H
