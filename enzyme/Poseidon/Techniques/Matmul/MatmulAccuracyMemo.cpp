//=- MatmulAccuracyMemo.cpp - Stored results of the matrix accuracy model -=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MatmulAccuracyMemo.h"

#include "BuildIdentity.h"
#include "Matmul.h"
#include "Pass/Session.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

// Exact, and readable back by StringRef::getAsDouble, infinity included.
static std::string accNumber(double v) {
  std::string s;
  raw_string_ostream(s) << format("%.17g", v);
  return s;
}

template <typename T> static void appendBytes(std::string &bytes, T v) {
  bytes.append(reinterpret_cast<const char *>(&v), sizeof(v));
}

static void addModelInputs(const Options &O, CacheKey &key,
                           const AbstractMatmul &m, const MatmulProfile &prof,
                           double confidence, unsigned sampleLogBits,
                           unsigned numSamples) {
  key.add("M", std::to_string(m.M));
  key.add("N", std::to_string(m.N));
  key.add("K", std::to_string(m.K));
  std::string operands;
  for (const std::vector<CellStat> *cells : {&prof.a, &prof.b}) {
    appendBytes(operands, (uint64_t)cells->size());
    for (const CellStat &c : *cells) {
      appendBytes(operands, c.min);
      appendBytes(operands, c.max);
      appendBytes(operands, c.count);
      appendBytes(operands, c.minMag);
    }
  }
  key.addContent("operand statistics", operands);
  std::string gradients;
  appendBytes(gradients, (uint64_t)prof.gradD.size());
  for (const GradCellStat &g : prof.gradD)
    appendBytes(gradients, g.sumGrad);
  key.addContent("gradient statistics", gradients);
  key.add("samples", std::to_string(numSamples));
  key.add("seed", std::to_string(O.Model.RandomSeed));
  key.add("sampling log bits", std::to_string(sampleLogBits));
  key.add("exponent penalty", accNumber(O.Model.ExponentPenalty));
  key.add("confidence", accNumber(confidence));
}

static std::string entryPath(const Options &O, const CacheKey &key) {
  SmallString<256> p(O.Herbie.Cache);
  sys::path::append(p, "matmulacc_" + key.digest() + ".json");
  return p.str().str();
}

// The name the memo had while it was keyed on a hash of the inputs alone, kept
// so -poseidon-import-accuracy-memo can bring its entries over.
namespace {
struct LegacyAccHash {
  uint64_t h = 1469598103934665603ULL;
  void mix(uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); }
  void mixD(double d) {
    uint64_t b;
    std::memcpy(&b, &d, sizeof(b));
    mix(b);
  }
  LegacyAccHash(const Options &O, const AbstractMatmul &m,
                const MatmulProfile &prof, unsigned sampleLogBits) {
    mix(m.M);
    mix(m.N);
    mix(m.K);
    for (const std::vector<CellStat> *cells : {&prof.a, &prof.b}) {
      mix(cells->size());
      for (const auto &c : *cells) {
        mixD(c.min);
        mixD(c.max);
        mixD(c.sumValue);
        mix(c.count);
        mixD(c.minMag);
      }
    }
    mix(prof.gradD.size());
    for (const auto &g : prof.gradD) {
      mixD(g.sumGrad);
      mix(g.count);
    }
    mix(O.Model.NumSamples);
    mix((uint64_t)O.Model.RandomSeed);
    mix((uint64_t)sampleLogBits);
    mix(1); // the model generation every legacy entry was written under
  }
  std::string name() const { return "matmulacc_" + utohexstr(h) + ".txt"; }
};
} // namespace

MatmulAccuracyMemo MatmulAccuracyMemo::matmul(
    const Options &O, const AbstractMatmul &m, const MatmulProfile &prof,
    double confidence, unsigned sampleLogBits, unsigned numSamples,
    FPKind inputPrec, FPKind accPrec, FPKind exponentPrec, unsigned orderTileK,
    unsigned inputMantBits) {
  MatmulAccuracyMemo memo;
  memo.key.add("model", "getMatmulAccuracyCost");
  memo.key.add("input format", fpKindName(inputPrec));
  memo.key.add("accumulator format", fpKindName(accPrec));
  memo.key.add("exponent format", fpKindName(exponentPrec));
  memo.key.add("orderTileK", std::to_string(orderTileK));
  memo.key.add("inputMantBits", std::to_string(inputMantBits));
  addModelInputs(O, memo.key, m, prof, confidence, sampleLogBits, numSamples);
  memo.path = entryPath(O, memo.key);
  memo.key.add("accuracy model", kAccuracyModelIdentity);
  if (!O.Herbie.ImportAccuracyMemo.empty()) {
    LegacyAccHash legacy(O, m, prof, sampleLogBits);
    legacy.mix(0x6d6dULL);
    legacy.mix((uint64_t)inputPrec);
    legacy.mix((uint64_t)accPrec);
    legacy.mix((uint64_t)exponentPrec);
    legacy.mixD(O.Model.ExponentPenalty);
    if (confidence != kDefaultConfidence)
      legacy.mixD(confidence);
    if (orderTileK)
      legacy.mix(0x746b00ULL | orderTileK);
    if (inputMantBits)
      legacy.mix(0x6d6200ULL | inputMantBits);
    memo.legacyName = legacy.name();
  }
  return memo;
}

MatmulAccuracyMemo
MatmulAccuracyMemo::ozakiII(const Options &O, const AbstractMatmul &m,
                            const MatmulProfile &prof, double confidence,
                            unsigned sampleLogBits, unsigned numSamples,
                            unsigned capturedBits) {
  MatmulAccuracyMemo memo;
  memo.key.add("model", "getOzakiIIAccuracyCost");
  memo.key.add("capturedBits", std::to_string(capturedBits));
  addModelInputs(O, memo.key, m, prof, confidence, sampleLogBits, numSamples);
  memo.path = entryPath(O, memo.key);
  memo.key.add("accuracy model", kAccuracyModelIdentity);
  if (!O.Herbie.ImportAccuracyMemo.empty()) {
    LegacyAccHash legacy(O, m, prof, sampleLogBits);
    legacy.mix(0x6f7aULL);
    legacy.mix(capturedBits);
    if (confidence != kDefaultConfidence)
      legacy.mixD(confidence);
    memo.legacyName = legacy.name();
  }
  return memo;
}

// No entry from before the memo kept its keys was computed by this model.
MatmulAccuracyMemo
MatmulAccuracyMemo::tcecLibrary(const Options &O, const AbstractMatmul &m,
                                const MatmulProfile &prof, double confidence,
                                unsigned sampleLogBits, unsigned numSamples,
                                TcecBackend backend) {
  MatmulAccuracyMemo memo;
  memo.key.add("model", "getTcecLibraryAccuracyCost");
  memo.key.add("TCEC backend", tcecBackendName(backend));
  addModelInputs(O, memo.key, m, prof, confidence, sampleLogBits, numSamples);
  memo.path = entryPath(O, memo.key);
  memo.key.add("accuracy model", kAccuracyModelIdentity);
  return memo;
}

void MatmulAccuracyMemo::store(const Options &O, double accuracyCost,
                               double domainError) const {
  if (O.Herbie.Cache.empty())
    return;
  (void)sys::fs::create_directories(O.Herbie.Cache, true);
  // As text: a domain error can be infinite, which JSON cannot hold.
  json::Object value;
  value["accuracy cost"] = accNumber(accuracyCost);
  value["domain error"] = accNumber(domainError);
  if (Error E = writeCacheEntry(path, key, std::move(value)))
    llvm::errs() << "[poseidon] could not write the accuracy memo " << path
                 << ": " << toString(std::move(E)) << "\n";
}

static bool parseAccNumber(const json::Object &value, StringRef name,
                           double &out) {
  std::optional<StringRef> text = value.getString(name);
  return text && !text->getAsDouble(out);
}

// Results stored before entries carried their key are read only when
// imported; a cache full of them is otherwise recomputed without a word, at
// the cost they were kept to save.
static void noteLegacyEntries(Session &S) {
  const Options &O = S.Opts;
  if (S.Matmul.LegacyMemoEntriesNoted)
    return;
  S.Matmul.LegacyMemoEntriesNoted = true;
  std::error_code EC;
  for (sys::fs::directory_iterator it(O.Herbie.Cache, EC), end;
       it != end && !EC; it.increment(EC)) {
    StringRef name = sys::path::filename(it->path());
    if (name.starts_with("matmulacc_") && name.ends_with(".txt")) {
      llvm::errs() << "[poseidon] " << O.Herbie.Cache
                   << " holds matrix accuracy results stored before the memo "
                      "kept their keys (matmulacc_*.txt), which are not read; "
                      "computing them again. -poseidon-import-accuracy-memo="
                   << O.Herbie.Cache << " takes them over instead.\n";
      return;
    }
  }
}

bool MatmulAccuracyMemo::lookup(Session &S, double &accuracyCost,
                                double &domainError) const {
  const Options &O = S.Opts;
  if (O.Herbie.Cache.empty())
    return false;
  CacheEntryRead read =
      readCacheEntry(path, key, {"accuracy model", kAccuracyModelEquivalents});
  if (read.value) {
    const json::Object *value = read.value->getAsObject();
    if (value && parseAccNumber(*value, "accuracy cost", accuracyCost) &&
        parseAccNumber(*value, "domain error", domainError))
      return true;
    read.unreadable = "its value does not parse";
  }
  if (!read.unreadable.empty())
    llvm::errs() << "[poseidon] accuracy memo " << path << " is unreadable ("
                 << read.unreadable << "); recomputing it.\n";
  else if (!read.differs.empty())
    llvm::errs() << "[poseidon] accuracy memo " << path
                 << (read.differs == "format"
                         ? " was written in another cache format"
                         : " was computed for a different " + read.differs)
                 << "; recomputing it.\n";

  if (O.Herbie.ImportAccuracyMemo.empty()) {
    noteLegacyEntries(S);
    return false;
  }
  if (legacyName.empty())
    return false;
  SmallString<256> legacy(O.Herbie.ImportAccuracyMemo);
  sys::path::append(legacy, legacyName);
  auto buf = MemoryBuffer::getFile(legacy);
  if (!buf)
    return false;
  auto [accS, domS] = (*buf)->getBuffer().trim().split(' ');
  double a, d;
  if (accS.getAsDouble(a) || domS.getAsDouble(d))
    return false;
  accuracyCost = a;
  domainError = d;
  store(O, accuracyCost, domainError);
  llvm::errs() << "[poseidon] imported accuracy memo " << legacy << " as "
               << path << "\n";
  return true;
}

} // namespace poseidon
