//==- MatmulAccuracyMemo.cpp - Stored results of the matrix accuracy model -==//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MatmulAccuracyMemo.h"

#include "Core/Options.h"
#include "Matmul.h"
#include "Sites/Annotations.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

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
  key.add("exponent penalty", exactDouble(O.Model.ExponentPenalty));
  key.add("confidence", exactDouble(confidence));
}

static std::string entryPath(const Options &O, const CacheKey &key) {
  SmallString<256> p(O.Herbie.Cache);
  sys::path::append(p, "matmulacc_" + key.digest() + ".json");
  return p.str().str();
}

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
  memo.key.addCompiler();
  return memo;
}

MatmulAccuracyMemo
MatmulAccuracyMemo::ozakiII(const Options &O, const AbstractMatmul &m,
                            const MatmulProfile &prof, double confidence,
                            unsigned sampleLogBits, unsigned numSamples,
                            unsigned moduli) {
  MatmulAccuracyMemo memo;
  memo.key.add("model", "getOzakiIIAccuracyCost");
  memo.key.add("moduli", std::to_string(moduli));
  addModelInputs(O, memo.key, m, prof, confidence, sampleLogBits, numSamples);
  memo.path = entryPath(O, memo.key);
  memo.key.addCompiler();
  return memo;
}

MatmulAccuracyMemo
MatmulAccuracyMemo::tcecLibrary(const Options &O, const AbstractMatmul &m,
                                const MatmulProfile &prof, double confidence,
                                unsigned sampleLogBits, unsigned numSamples) {
  MatmulAccuracyMemo memo;
  memo.key.add("model", "getTcecLibraryAccuracyCost");
  // The library whose product the model simulates.
  memo.key.add("TCEC backend", "cuMpSGEMM");
  addModelInputs(O, memo.key, m, prof, confidence, sampleLogBits, numSamples);
  memo.path = entryPath(O, memo.key);
  memo.key.addCompiler();
  return memo;
}

void MatmulAccuracyMemo::store(const Options &O, double accuracyCost,
                               double domainError) const {
  if (O.Herbie.Cache.empty())
    return;
  (void)sys::fs::create_directories(O.Herbie.Cache, true);
  // As text: a domain error can be infinite, which JSON cannot hold.
  json::Object value;
  value["accuracy cost"] = exactDouble(accuracyCost);
  value["domain error"] = exactDouble(domainError);
  if (Error E = writeCacheEntry(path, key, std::move(value)))
    llvm::errs() << "[poseidon] could not write the accuracy memo " << path
                 << ": " << toString(std::move(E)) << '\n';
}

static bool parseAccNumber(const json::Object &value, StringRef name,
                           double &out) {
  std::optional<StringRef> text = value.getString(name);
  return text && !text->getAsDouble(out);
}

bool MatmulAccuracyMemo::lookup(const Options &O, double &accuracyCost,
                                double &domainError) const {
  if (O.Herbie.Cache.empty())
    return false;
  CacheEntryRead read = readCacheEntry(path, key);
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
  return false;
}

} // namespace poseidon
