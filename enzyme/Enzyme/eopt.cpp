#include "llvm/ADT/ArrayRef.h"
#include "llvm/Passes/PassBuilder.h"

#include <functional>

using namespace llvm;

void registerEnzyme(llvm::PassBuilder &PB);

extern "C" int optMain(int argc, char **argv,
                       llvm::ArrayRef<std::function<void(llvm::PassBuilder &)>>
                           PassBuilderCallbacks);

int main(int argc, char **argv) {
  std::function<void(llvm::PassBuilder &)> plugins[] = {registerEnzyme};
  return optMain(argc, argv, plugins);
}
