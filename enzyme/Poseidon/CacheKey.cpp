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
CacheKey::firstDifference(const json::Object *stored) const {
  if (!stored)
    return std::string("format");
  for (const auto &[name, value] : parts) {
    std::optional<StringRef> got = stored->getString(name);
    if (!got || *got != value)
      return name;
  }
  if (stored->size() != parts.size())
    for (const auto &kv : *stored)
      if (llvm::none_of(parts,
                        [&](const auto &p) { return p.first == kv.first; }))
        return kv.first.str();
  return std::nullopt;
}

} // namespace poseidon
