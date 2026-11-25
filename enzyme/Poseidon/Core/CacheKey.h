//===- CacheKey.h - what a cached result was computed from ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Poseidon's caches, Herbie's results, the per-site DP tables and the matrix
// accuracy memo, are keyed on the inputs the cached result was computed from,
// as named parts. A Herbie result is filed under the digest of its key. A DP
// table entry and an accuracy memo entry store their key's parts, so a
// mismatch can say which input changed.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CACHEKEY_H
#define POSEIDON_CACHEKEY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>
#include <utility>

namespace poseidon {

class CacheKey {
public:
  // The version of the layout results are stored in; every result written
  // under another format is a miss.
  static constexpr unsigned kFormat = 1;

  CacheKey();
  void add(llvm::StringRef name, llvm::StringRef value);
  // Part "compiler": this build of the pass (the digest of its sources) and the
  // LLVM it is built against, so that a result another build computed misses.
  void addCompiler();
  // For a part too large to store, such as a file: kept as its digest.
  void addContent(llvm::StringRef name, llvm::StringRef bytes);

  std::string digest() const;
  llvm::json::Object toJSON() const;
  // The name of the first part `stored` disagrees on, "format" when it was
  // written under another format or under none; nullopt when it agrees. A part
  // named in `unchecked` agrees whatever `stored` holds for it.
  std::optional<std::string>
  firstDifference(const llvm::json::Object *stored,
                  llvm::ArrayRef<llvm::StringRef> unchecked = {}) const;

private:
  llvm::SmallVector<std::pair<std::string, std::string>, 64> parts;
};

std::string contentDigest(llvm::StringRef bytes);

// Replaces `path` by `contents` through a temporary file in its directory, so
// a reader sees the old file or the new one, never part of either.
llvm::Error writeFileAtomically(llvm::StringRef path, llvm::StringRef contents);

// A cache with one file per entry holds the entry's key and its value as JSON.
llvm::Error writeCacheEntry(llvm::StringRef path, const CacheKey &key,
                            llvm::json::Value value);

struct CacheEntryRead {
  // The stored value, when the stored key equals `key` in every part but those
  // named in `unchecked`.
  std::optional<llvm::json::Value> value;
  // Otherwise the first part it differs in (see firstDifference), or why the
  // file is not an entry; both empty when there is no file.
  std::string differs;
  std::string unreadable;
};
CacheEntryRead readCacheEntry(llvm::StringRef path, const CacheKey &key,
                              llvm::ArrayRef<llvm::StringRef> unchecked = {});

} // namespace poseidon
#endif // POSEIDON_CACHEKEY_H
