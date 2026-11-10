//=- CacheKey.cpp - what a cached result was computed from ----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CacheKey.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;

namespace poseidon {

std::string contentDigest(StringRef bytes) {
  return utohexstr(xxh3_64bits(bytes), /*LowerCase=*/false, /*Width=*/16);
}

CacheKey::CacheKey() { add("format", std::to_string(kFormat)); }

void CacheKey::add(StringRef name, StringRef value) {
  parts.emplace_back(name.str(), value.str());
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
                          const Equivalents &equivalents) const {
  if (!stored)
    return std::string("format");
  for (const auto &[name, value] : parts) {
    std::optional<StringRef> got = stored->getString(name);
    if (got && *got == value)
      continue;
    if (got && name == equivalents.part &&
        llvm::is_contained(equivalents.values, *got))
      continue;
    return name;
  }
  if (stored->size() != parts.size())
    for (const auto &kv : *stored)
      if (llvm::none_of(parts,
                        [&](const auto &p) { return p.first == kv.first; }))
        return kv.first.str();
  return std::nullopt;
}

Error writeFileAtomically(StringRef path, StringRef contents) {
  return writeToOutput(path, [&](raw_ostream &os) {
    os << contents;
    return Error::success();
  });
}

Error writeCacheEntry(StringRef path, const CacheKey &key, json::Value value) {
  json::Object entry;
  entry["key"] = key.toJSON();
  entry["value"] = std::move(value);
  return writeFileAtomically(
      path, formatv("{0:2}", json::Value(std::move(entry))).str() + "\n");
}

CacheEntryRead readCacheEntry(StringRef path, const CacheKey &key,
                              const CacheKey::Equivalents &equivalents) {
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
          key.firstDifference(entry->getObject("key"), equivalents)) {
    read.differs = *part;
    return read;
  }
  read.value = *value;
  return read;
}

} // namespace poseidon
