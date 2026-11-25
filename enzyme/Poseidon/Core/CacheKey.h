//===- CacheKey.h - what a cached result was computed from ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A cache key names the inputs a cached result was computed from, part by
// part. A cache can file a result under the key's digest, or store the key's
// parts beside the result in an entry, so that a mismatch says which input
// changed.
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

/// What a cached result was computed from: named parts, in the order added,
/// that an entry is stored and looked up under.
class CacheKey {
public:
  /// The version of the layout results are stored in; every result written
  /// under another format is a miss.
  static constexpr unsigned kFormat = 1;

  /// A key holding the part "format", kFormat.
  CacheKey();
  /// Appends the part `name` with `value`.
  void add(llvm::StringRef name, llvm::StringRef value);
  /// Part "compiler": `build`, which names the build of the program computing
  /// the result, and the LLVM it is built against, so that a result another
  /// build computed misses.
  void addCompiler(llvm::StringRef build);
  /// For a part too large to store, such as a file: kept as its digest.
  void addContent(llvm::StringRef name, llvm::StringRef bytes);

  /// A digest of every part, names and values in order, as 16 hex digits: what
  /// a cache file is named by.
  std::string digest() const;
  /// The parts as a JSON object, as an entry stores its key.
  llvm::json::Object toJSON() const;
  /// The name of the first part `stored` disagrees on, "format" when it was
  /// written under another format or under none; nullopt when it agrees. A part
  /// named in `unchecked` agrees whatever `stored` holds for it.
  std::optional<std::string>
  firstDifference(const llvm::json::Object *stored,
                  llvm::ArrayRef<llvm::StringRef> unchecked = {}) const;

private:
  /// The (name, value) parts, in the order added.
  llvm::SmallVector<std::pair<std::string, std::string>, 64> parts;
};

/// The 64-bit xxh3 digest of `bytes` as 16 upper-case hex digits.
std::string contentDigest(llvm::StringRef bytes);

/// Replaces `path` by `contents` through a temporary file in its directory, so
/// a reader sees the old file or the new one, never part of either.
llvm::Error writeFileAtomically(llvm::StringRef path, llvm::StringRef contents);

/// A cache with one file per entry holds the entry's key and its value as JSON.
llvm::Error writeCacheEntry(llvm::StringRef path, const CacheKey &key,
                            llvm::json::Value value);

/// What reading one cache entry found.
struct CacheEntryRead {
  /// The stored value, when the stored key equals `key` in every part but those
  /// named in `unchecked`.
  std::optional<llvm::json::Value> value;
  /// Otherwise the first part it differs in (see firstDifference), or why the
  /// file is not an entry; both empty when there is no file.
  std::string differs;
  /// Why the file is not an entry; empty when it is one or there is no file.
  std::string unreadable;
};
/// The entry at `path` read against `key`, the parts named in `unchecked` not
/// compared. A missing file leaves every field empty.
CacheEntryRead readCacheEntry(llvm::StringRef path, const CacheKey &key,
                              llvm::ArrayRef<llvm::StringRef> unchecked = {});

} // namespace poseidon
#endif // POSEIDON_CACHEKEY_H
