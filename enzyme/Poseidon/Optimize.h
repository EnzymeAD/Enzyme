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

bool isOptimizable(const llvm::Value &V);

bool demoteFPCastPHIs(llvm::Function &F);

void setSlotMetadata(llvm::Function &F);
void preprocess(llvm::Function *F);
// `errorTol` > 0 is the accuracy target written AT the site; 0 means the site
// wrote none and -poseidon-tau decides.
bool fpOptimize(Session &S, llvm::Function &F, double errorTol = 0.0);

// Joint-DP entry: optimizeSiteBody marks each site with "poseidon-joint-errtol"
// instead of optimizing it inline; this solves one DP over all marked sites.
bool solveJointly(Session &S, llvm::Module &M,
                  llvm::FunctionAnalysisManager &FAM);

void noteSiteOrigin(Session &S, llvm::Function *clone, llvm::Function *orig);
const llvm::Function *siteOriginOf(const Session &S,
                                   const llvm::Function &clone);
// The ".fpprofile" stem of a site clone. The profiling run writes one record
// per marked BODY, so every clone of one body reads the same file.
std::string siteProfileStem(const Session &S, const llvm::Function &clone);
// The name -poseidon-apply-rewrites gives a site by: the symbol its body was
// cut from (an annotated kernel or host function, a marker's target, the copy
// made for one __poseidon_optimize call), the one its profile is named after.
std::string siteKeyOfBody(const llvm::Function &body);
std::string siteKey(const Session &S, const llvm::Function &clone);
// The other names a site answers to when no other site of the module does:
// the function a per-call copy was made of, and the unqualified name the
// symbol demangles to.
llvm::SmallVector<std::string, 2> siteAliases(llvm::StringRef key);
// Caller value of parameter i of a site clone, as recorded by
// noteSiteArgs; null when unknown.
llvm::Value *siteArg(const Session &S, const llvm::Function *clone, unsigned i);
// Whether noteSiteArgs recorded F: a site clone, not a pricing clone.
bool isSiteClone(const Session &S, const llvm::Function *F);
void noteSiteArgs(Session &S, llvm::Function *clone,
                  llvm::ArrayRef<llvm::Value *> primalArgs);
bool redirectNoopSite(Session &S, llvm::Function *clone);

} // namespace poseidon
#endif // POSEIDON_OPTIMIZE_H
