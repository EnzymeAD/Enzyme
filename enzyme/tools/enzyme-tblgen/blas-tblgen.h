#include "llvm/ADT/SmallString.h"
#include "llvm/TableGen/Record.h"

class TGPattern;

void emitBlasDerivatives(const llvm::RecordKeeper &RK, llvm::raw_ostream &os);
bool hasDiffeRet(llvm::Init *resultTree);
bool hasAdjoint(TGPattern &pattern, llvm::Init *resultTree,
                llvm::StringRef argName);
llvm::SmallString<80> ValueType_helper(const TGPattern &pattern,
                                       ssize_t actPos);