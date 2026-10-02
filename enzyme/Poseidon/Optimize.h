#ifndef POSEIDON_OPTIMIZE_H
#define POSEIDON_OPTIMIZE_H

#include <string>

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace poseidon {

bool isOptimizable(const llvm::Value &V);

bool demoteFPCastPHIs(llvm::Function &F);

void setSlotMetadata(llvm::Function &F);
void preprocess(llvm::Function *F);
// `siteConfidence` > 0 is the confidence level written AT the site; 0 means
// the site wrote none and -poseidon-confidence decides.
bool fpOptimize(llvm::Function &F, double errorTol = 0.0,
                double siteConfidence = 0.0);

// Joint-DP entry: optimizeSiteBody marks each site with "poseidon-joint-errtol"
// instead of optimizing it inline; this solves one DP over all marked sites.
bool solveJointly(llvm::Module &M, llvm::FunctionAnalysisManager &FAM);

void noteSiteOrigin(llvm::Function *clone, llvm::Function *orig);
// The ".fpprofile" stem of a site clone. The profiling run writes one record
// per marked BODY, so every clone of one body reads the same file.
std::string siteProfileStem(const llvm::Function &clone);
// Caller-supplied primal value of parameter i of a site clone, as recorded by
// noteSiteArgs; null when unknown.
llvm::Value *siteArg(const llvm::Function *clone, unsigned i);

void noteSiteArgs(llvm::Function *clone,
                  llvm::ArrayRef<llvm::Value *> primalArgs);
bool redirectNoopSite(llvm::Function *clone);

} // namespace poseidon
#endif // POSEIDON_OPTIMIZE_H
