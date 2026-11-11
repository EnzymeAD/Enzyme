#ifndef POSEIDON_OPTIMIZE_H
#define POSEIDON_OPTIMIZE_H

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace poseidon {

class Session;

// `errorTol` > 0 is the accuracy target written AT the site; 0 means the site
// wrote none and -poseidon-tau decides.
bool fpOptimize(Session &S, llvm::Function &F, double errorTol = 0.0);

// Joint-DP entry: optimizeSiteBody marks each site with "poseidon-joint-errtol"
// instead of optimizing it inline; this solves one DP over all marked sites.
bool solveJointly(Session &S, llvm::Module &M,
                  llvm::FunctionAnalysisManager &FAM);

} // namespace poseidon
#endif // POSEIDON_OPTIMIZE_H
