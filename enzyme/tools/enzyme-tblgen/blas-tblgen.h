#include "llvm/ADT/SmallString.h"
#include "llvm/TableGen/Record.h"

class TGPattern;

void emitBlasDerivatives(const llvm::RecordKeeper &RK, llvm::raw_ostream &os);
bool hasDiffeRet(const llvm::Init *resultTree);
bool hasAdjoint(const TGPattern &pattern, const llvm::Init *resultTree,
                llvm::StringRef argName);
llvm::SmallString<80> ValueType_helper(const TGPattern &pattern, ssize_t actPos,
                                       const llvm::DagInit *ruleDag);
