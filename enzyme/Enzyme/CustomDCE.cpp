#include "CustomDCE.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"

using namespace llvm;

PreservedAnalyses CustomDCEPass::run(Function &F, FunctionAnalysisManager &AM) {
  errs() << "Hello, running on " << F.getName() << "\n";
  // TODO: Find a way to communicate const arguments here
  SmallDenseSet<unsigned> constArgs = {1};
  for (const auto &[i, arg] : llvm::enumerate(F.args())) {
    if (!constArgs.contains(i))
      continue;

    // Traverse def-use chains of the argument
    SmallVector<Instruction *> frontier;
    SetVector<Instruction *> visited;
    for (User *U : arg.users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        frontier.push_back(I);
      }
    };

    while (!frontier.empty()) {
      Instruction *I = frontier.back();
      frontier.pop_back();

      if (!visited.contains(I)) {
        visited.insert(I);
        for (User *U : I->users()) {
          if (auto *neighbor = dyn_cast<Instruction>(U)) {
            if (!visited.contains(neighbor))
              frontier.push_back(neighbor);
          }
        }
      }
    }

    // Delete any instruction that (transitively) uses the argument, starting
    // with the end of the def-use chains
    for (Instruction *I : llvm::reverse(visited)) {
      errs() << "Removing instruction " << *I << "\n";
      if (I->use_empty())
        I->eraseFromParent();
      else {
        errs() << "[custom-dce] could not remove inst because it had users: "
               << *I << "\n";
      }
    }
  }
  return PreservedAnalyses::all();
}
