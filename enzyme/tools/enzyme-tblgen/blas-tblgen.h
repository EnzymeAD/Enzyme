#include "llvm/ADT/SmallString.h"
#include "llvm/TableGen/Record.h"

class TGPattern;

void emitBlasDerivatives(const llvm::RecordKeeper &RK, llvm::raw_ostream &os);
bool hasDiffeRet(llvm::Init *resultTree);
bool hasAdjoint(llvm::Init *resultTree, llvm::StringRef argName);
llvm::SmallString<80> ValueType_helper(const TGPattern &pattern, size_t actPos);