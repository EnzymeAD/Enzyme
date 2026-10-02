// Declarations shared by the pieces of the matrix-product pipeline (the
// Matmul*.cpp and RecognizeHostGemm.cpp sources).
#ifndef POSEIDON_MATMUL_INTERNAL_H
#define POSEIDON_MATMUL_INTERNAL_H

#include "Matmul.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

#include <cstdint>
#include <string>

namespace poseidon {

std::string mmaShapeSuffix(unsigned M, unsigned N, unsigned K);
double getMatmulCompCost(unsigned M, unsigned N, unsigned K, FPKind inputPrec,
                         FPKind accPrec);
std::string inKernelDirectClass(FPKind inputPrec, FPKind accPrec);
std::string inKernelTcecClass(unsigned n, FPKind inputPrec, FPKind accPrec);
bool priceInKernelRaiseFromMeasuredRow(const AbstractMatmul &m,
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
const llvm::SmallVector<WmmaTarget, 16> &getAvailableWmmaTargets();
bool inKernelRaiseFits(llvm::Function *F, const AbstractMatmul &m,
                       const CandidateMatmul::Option &opt,
                       uint64_t existingBytes, llvm::StringRef className);

// `confidence` is the fraction of the sampled inputs the returned domain error
// bounds (the percentile it is read off), the site's own or
// -poseidon-confidence.
double getMatmulAccuracyCost(const AbstractMatmul &m, const MatmulProfile &prof,
                             double confidence, unsigned sampleLogBits,
                             FPKind inputPrec, FPKind accPrec,
                             double *domainErrOut = nullptr,
                             FPKind exponentPrec = FPKind::Invalid,
                             unsigned orderTileK = 0,
                             unsigned inputMantBits = 0);
double getOzakiIIAccuracyCost(const AbstractMatmul &m,
                              const MatmulProfile &prof, double confidence,
                              unsigned sampleLogBits, unsigned capturedBits,
                              double *domainErrOut = nullptr);

} // namespace poseidon
#endif // POSEIDON_MATMUL_INTERNAL_H
