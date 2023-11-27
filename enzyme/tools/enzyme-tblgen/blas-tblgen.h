#include <llvm/TableGen/Record.h>

void emitBlasDerivatives(const llvm::RecordKeeper &RK, llvm::raw_ostream &os);
bool hasDiffeRet(llvm::Init *resultTree);
bool hasAdjoint(llvm::Init *resultTree, llvm::StringRef argName);
