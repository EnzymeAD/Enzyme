//===- CacheKey.cpp - what a cached result was computed from --------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CacheKey.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#if __has_include("llvm/Support/VCSRevision.h")
#include "llvm/Support/VCSRevision.h"
#endif

using namespace llvm;

namespace poseidon {

std::string contentDigest(StringRef bytes) {
  return utohexstr(xxh3_64bits(bytes), /*LowerCase=*/false, /*Width=*/16);
}

CacheKey::CacheKey() { add("format", std::to_string(kFormat)); }

void CacheKey::add(StringRef name, StringRef value) {
  parts.emplace_back(name.str(), value.str());
}

void CacheKey::addCompiler(StringRef build) {
  // Two LLVM snapshots report one version string; the revision, which an LLVM
  // built without it does not define, tells them apart.
  std::string compiler = build.str() + " LLVM " + LLVM_VERSION_STRING;
#ifdef LLVM_REVISION
  compiler += " " LLVM_REVISION;
#endif
  add("compiler", compiler);
}

void CacheKey::addContent(StringRef name, StringRef bytes) {
  add(name, contentDigest(bytes));
}

std::string CacheKey::digest() const {
  std::string joined;
  for (const auto &[name, value] : parts) {
    joined += name;
    joined += '\0';
    joined += value;
    joined += '\0';
  }
  return contentDigest(joined);
}

json::Object CacheKey::toJSON() const {
  json::Object obj;
  for (const auto &[name, value] : parts)
    obj[name] = value;
  return obj;
}

std::optional<std::string>
CacheKey::firstDifference(const json::Object *stored,
                          ArrayRef<StringRef> unchecked) const {
  if (!stored)
    return std::string("format");
  for (const auto &[name, value] : parts)
    if (!is_contained(unchecked, name) &&
        stored->getString(name) != StringRef(value))
      return name;
  // By name: the object holds its keys in hash order.
  for (const json::Object::value_type *kv : json::sortedElements(*stored))
    if (!is_contained(unchecked, StringRef(kv->first)) &&
        none_of(parts, [&](const auto &p) { return p.first == kv->first; }))
      return kv->first.str();
  return std::nullopt;
}

Error writeCacheEntry(StringRef path, const CacheKey &key, json::Value value) {
  if (std::error_code EC =
          sys::fs::create_directories(sys::path::parent_path(path)))
    return errorCodeToError(EC);
  json::Object entry;
  entry["key"] = key.toJSON();
  entry["value"] = std::move(value);
  return writeToOutput(path, [&](raw_ostream &os) {
    os << formatv("{0:2}", json::Value(std::move(entry))) << "\n";
    return Error::success();
  });
}

CacheEntryRead readCacheEntry(StringRef path, const CacheKey &key,
                              ArrayRef<StringRef> unchecked) {
  CacheEntryRead read;
  if (!sys::fs::exists(path))
    return read;
  auto buf = MemoryBuffer::getFile(path);
  if (!buf) {
    read.unreadable = buf.getError().message();
    return read;
  }
  Expected<json::Value> root = json::parse((*buf)->getBuffer());
  if (!root) {
    read.unreadable = toString(root.takeError());
    return read;
  }
  json::Object *entry = root->getAsObject();
  const json::Value *value = entry ? entry->get("value") : nullptr;
  if (!value) {
    read.unreadable = "it holds no entry";
    return read;
  }
  if (std::optional<std::string> part =
          key.firstDifference(entry->getObject("key"), unchecked)) {
    read.differs = *part;
    return read;
  }
  read.value = *value;
  return read;
}

} // namespace poseidon
