// Declarations shared by the pieces of the matrix-product pipeline (the
// Matmul*.cpp and RecognizeHostGemm.cpp sources).
#ifndef POSEIDON_MATMUL_INTERNAL_H
#define POSEIDON_MATMUL_INTERNAL_H

#include "HostDispatch.h"
#include "Matmul.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>

namespace poseidon {

struct CostModelState;
struct Options;
class Session;

// An instruction of the product, where an error about it is reported.
inline const llvm::Instruction &anchorOf(const AbstractMatmul &m) {
  if (m.scalarLoop.fma)
    return *m.scalarLoop.fma;
  if (m.hostGemm && m.hostGemm->cStore)
    return *m.hostGemm->cStore;
  return *m.footprint.front();
}

std::string mmaShapeSuffix(unsigned M, unsigned N, unsigned K);
std::string inKernelCostClass(const SchemeVariant &v);
bool priceInKernelRaiseFromMeasuredRow(Session &S, const AbstractMatmul &m,
                                       const std::string &cls, unsigned tileM,
                                       unsigned tileN, unsigned tileK,
                                       double baselinePerMac,
                                       const std::string &what,
                                       double &costOut);

struct WmmaTarget {
  unsigned M, N, K;
  FPKind inputPrec;
  FPKind accPrec;
};
const llvm::SmallVector<WmmaTarget, 16> &getAvailableWmmaTargets(Session &S);

// What the matrix-product pipeline records over a compile.
struct MatmulState {
  // HostDispatch.cpp. fpOptimize records the note keyed by the optimized body
  // clone; optimizeSiteBody retrieves it after fpOptimize returns.
  NoteMap<GemmBodyNote> GemmBodies;
  // In the order the sites were split, so that of two sites of one wrapper the
  // same one is recorded last in every compile.
  llvm::MapVector<const llvm::Function *, PendingGemm> PendingGemms;

  // The tensor-core shapes the cost model prices, and the warnings already
  // given.
  std::optional<llvm::SmallVector<WmmaTarget, 16>> AvailableWmmaTargets;
  std::set<const void *> WarnedRectangularRows;
  std::set<const void *> WarnedSquareNativeRows;
  std::set<std::string> ReportedDirectRows;
  std::set<std::string> ReportedInKernelRows;
  bool LegacyMemoEntriesNoted = false;
};
bool inKernelRaiseFits(const Options &O, llvm::Function *F,
                       const AbstractMatmul &m, const MatmulUnit::Option &opt,
                       uint64_t existingBytes, llvm::StringRef className);

// `confidence` is the fraction of the sampled inputs the returned domain error
// bounds (the percentile it is read off).
double getMatmulAccuracyCost(Session &S, const AbstractMatmul &m,
                             const MatmulProfile &prof, double confidence,
                             unsigned sampleLogBits, FPKind inputPrec,
                             FPKind accPrec, double *domainErrOut = nullptr,
                             FPKind exponentPrec = FPKind::Invalid,
                             unsigned orderTileK = 0,
                             unsigned inputMantBits = 0);
// __poseidon_tcec_dgemm in compute `mode`, on cuMpSGEMM.
double getTcecLibraryAccuracyCost(Session &S, const AbstractMatmul &m,
                                  const MatmulProfile &prof, double confidence,
                                  unsigned sampleLogBits, unsigned mode,
                                  double *domainErrOut = nullptr);
double getOzakiIIAccuracyCost(Session &S, const AbstractMatmul &m,
                              const MatmulProfile &prof, double confidence,
                              unsigned sampleLogBits, unsigned moduli,
                              double *domainErrOut = nullptr);

} // namespace poseidon
#endif // POSEIDON_MATMUL_INTERNAL_H
