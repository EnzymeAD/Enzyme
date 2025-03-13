#include "CustomDCE.h"
#include "Utils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static llvm::cl::opt<std::string>
    DCEFunc("custom-dce-func", cl::init(""), cl::Hidden,
            cl::desc("Function to perform DCE on"));

static llvm::cl::opt<std::string>
    DCEArgs("custom-dce-indices", cl::init(""), cl::Hidden,
            cl::desc("Comma-separated indices of arguments to DCE"));

PreservedAnalyses CustomDCEPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (F.getName() != DCEFunc)
    return PreservedAnalyses::all();

  std::vector<ssize_t> indices = getDCEIndices(DCEArgs);
  if (indices.empty())
    return PreservedAnalyses::all();

  SmallDenseSet<unsigned> constArgs(indices.begin(), indices.end());
  SetVector<Instruction *> visited;
  for (const auto &[i, arg] : llvm::enumerate(F.args())) {
    if (!constArgs.contains(i))
      continue;

    // Traverse def-use chains of the argument
    SmallVector<Instruction *> frontier;
    for (User *U : arg.users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        frontier.push_back(I);
      }
    };

    while (!frontier.empty()) {
      Instruction *I = frontier.back();
      frontier.pop_back();
      if (isa<CallInst>(I))
        continue;
      else if (!visited.contains(I)) {
        visited.insert(I);
        for (User *U : I->users()) {
          if (auto *neighbor = dyn_cast<Instruction>(U)) {
            if (!visited.contains(neighbor))
              frontier.push_back(neighbor);
          }
        }
      }
    }
  }

  // Delete any instruction that (transitively) uses the argument, starting
  // with the end of the def-use chains to hopefully reduce the number of
  // iterations
  ssize_t found = visited.size();
  ssize_t removed = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (Instruction *I : llvm::reverse(visited)) {
      errs() << "    " << *I;
      if (I->use_empty()) {
        I->eraseFromParent();
        visited.remove(I);
        changed = true;
        removed++;
      } else {
        errs() << " [did not remove, had users]";
      }
      errs() << "\n";
    }
  }
  errs() << "[custom-dce] when processing function " << F.getName() << " ("
         << F.arg_size() << " args)"
         << "\n"
         << "[custom-dce] found " << found
         << " directly dependent instructions, removed " << removed << "\n";
  return PreservedAnalyses::none();
}
