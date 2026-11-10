// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang++ $(llvm-config --cxxflags) -I%S/../../Poseidon %S/../../Poseidon/Core/CacheKey.cpp %s $(llvm-config --ldflags --libs support --system-libs) -o %t/cache_key
// RUN: %t/cache_key %t | FileCheck %s
// RUN: FileCheck --check-prefix=ENTRY %s < %t/model.json

// An entry is a hit only when its stored key equals the computed one in every
// part, except that a part may hold one of the earlier values declared
// equivalent for that part, and only for that part.

#include "Core/CacheKey.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace poseidon;

static CacheKey key(StringRef seed, StringRef model) {
  CacheKey k;
  k.add("seed", seed);
  k.add("accuracy model", model);
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
  const std::string model = std::string(argv[1]) + "/model.json";
  const std::string seed = std::string(argv[1]) + "/seed.json";
  static const char *const earlier[] = {"0123456789ABCDEF"};
  const CacheKey::Equivalents equivalents{"accuracy model", earlier};

  show("absent", readCacheEntry(model, key("7", "0123456789ABCDEF")));
  if (Error E = writeCacheEntry(model, key("7", "0123456789ABCDEF"), 42))
    return errs() << toString(std::move(E)) << "\n", 1;
  if (Error E = writeCacheEntry(
          seed, key("0123456789ABCDEF", "FEDCBA9876543210"), 43))
    return errs() << toString(std::move(E)) << "\n", 1;

  show("equal", readCacheEntry(model, key("7", "0123456789ABCDEF")));
  show("seed differs",
       readCacheEntry(model, key("8", "0123456789ABCDEF"), equivalents));
  show("model differs, unlisted",
       readCacheEntry(model, key("7", "FEDCBA9876543210")));
  show("model differs, listed",
       readCacheEntry(model, key("7", "FEDCBA9876543210"), equivalents));
  show("listed value in another part",
       readCacheEntry(seed, key("7", "FEDCBA9876543210"), equivalents));
  return 0;
}

// CHECK: absent: no entry
// CHECK-NEXT: equal: hit 42
// CHECK-NEXT: seed differs: miss, different seed
// CHECK-NEXT: model differs, unlisted: miss, different accuracy model
// CHECK-NEXT: model differs, listed: hit 42
// CHECK-NEXT: listed value in another part: miss, different seed

// ENTRY: "key": {
// ENTRY-NEXT: "accuracy model": "0123456789ABCDEF",
// ENTRY-NEXT: "format": "1",
// ENTRY-NEXT: "seed": "7"
// ENTRY-NEXT: },
// ENTRY-NEXT: "value": 42
