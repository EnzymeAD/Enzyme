// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_test_programs/poseidon-test-cache-key %t | FileCheck %s
// RUN: FileCheck --check-prefix=ENTRY %s < %t/entry.json

// An entry is a hit only when its stored key equals the computed one in every
// part; a miss names the first part that differs.

#include "Core/CacheKey.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace poseidon;

static CacheKey key(StringRef seed, StringRef compiler) {
  CacheKey k;
  k.add("seed", seed);
  k.add("compiler", compiler);
  return k;
}

static void show(StringRef what, const CacheEntryRead &read) {
  outs() << what << ": ";
  if (read.value)
    outs() << "hit " << *read.value;
  else if (!read.differs.empty())
    outs() << "miss, different " << read.differs;
  else if (!read.unreadable.empty())
    outs() << "unreadable";
  else
    outs() << "no entry";
  outs() << "\n";
}

int main(int argc, char **argv) {
  const std::string entry = std::string(argv[1]) + "/entry.json";

  show("absent", readCacheEntry(entry, key("7", "0123456789ABCDEF")));
  if (Error E = writeCacheEntry(entry, key("7", "0123456789ABCDEF"), 42))
    return errs() << toString(std::move(E)) << "\n", 1;

  show("equal", readCacheEntry(entry, key("7", "0123456789ABCDEF")));
  show("seed differs", readCacheEntry(entry, key("8", "0123456789ABCDEF")));
  show("compiler differs", readCacheEntry(entry, key("7", "FEDCBA9876543210")));
  return 0;
}

// CHECK: absent: no entry
// CHECK-NEXT: equal: hit 42
// CHECK-NEXT: seed differs: miss, different seed
// CHECK-NEXT: compiler differs: miss, different compiler

// ENTRY: "key": {
// ENTRY-NEXT: "compiler": "0123456789ABCDEF",
// ENTRY-NEXT: "format": "1",
// ENTRY-NEXT: "seed": "7"
// ENTRY-NEXT: },
// ENTRY-NEXT: "value": 42
