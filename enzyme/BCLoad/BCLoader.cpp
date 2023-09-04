#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <string>

using namespace llvm;

#define DATA EnzymeBlasBC
#include "blas_headers.h"
#undef DATA

bool provideDefinitions(Module &M, std::set<std::string> ignoreFunctions = {}) {
  std::vector<StringRef> todo;
  bool seen32 = false;
  bool seen64 = false;
  bool seenGemm = false;
  for (auto &F : M) {
    if (!F.empty())
      continue;
    int index = 0;
    for (auto postfix : {"", "_", "_64_"}) {
      std::string str;
      if (strlen(postfix) == 0) {
        str = F.getName().str();
        if (ignoreFunctions.count(str)) continue;
      } else if (F.getName().endswith(postfix)) {
        auto blasName =
            F.getName().substr(0, F.getName().size() - strlen(postfix)).str();
        if (ignoreFunctions.count(blasName)) continue;
        str = "cblas_" + blasName;
      }

      auto found = EnzymeBlasBC.find(str);
      if (found != EnzymeBlasBC.end()) {
        todo.push_back(found->second);
        if (index == 1)
          seen32 = true;
        if (index == 2)
          seen64 = true;
        if (StringRef(str).endswith("gemm"))
          seenGemm = true;
        break;
      }
      index++;
    }
  }

  // Push fortran wrapper libs before all the other blas
  // to ensure the fortran injections have their code
  // replaced
  if (seen32)
    todo.insert(todo.begin(), __data_fblas32);
  if (seen64)
    todo.insert(todo.begin(), __data_fblas64);
  if (seenGemm) {
    todo.push_back(__data_xerbla);
  }
  bool changed = false;
  for (auto mod : todo) {
    SMDiagnostic Err;
    MemoryBufferRef buf(mod, StringRef("bcloader"));

    auto BC = llvm::parseIR(buf, Err, M.getContext(), [&](StringRef) {
      return Optional<std::string>(M.getDataLayout().getStringRepresentation());
    });

    if (!BC)
      Err.print("bcloader", llvm::errs());
    assert(BC);
    SmallVector<std::string, 1> toReplace;
    for (auto &F : *BC) {
      if (F.empty())
        continue;
      toReplace.push_back(F.getName().str());
    }
    BC->setTargetTriple("");
    Linker L(M);
    L.linkInModule(std::move(BC));
    for (auto name : toReplace) {
      if (auto F = M.getFunction(name)) {
        F->setLinkage(Function::LinkageTypes::InternalLinkage);
        F->addFnAttr(Attribute::AlwaysInline);
      }
    }
    changed = true;
  }
  return changed;
}

extern "C" {
uint8_t EnzymeBitcodeReplacement(LLVMModuleRef M, char **FncsNamesToIgnore,
                                 size_t numFncNames) {
  std::set<std::string> ignoreFunctions = {};
  for (size_t i = 0; i < numFncNames; i++) {
    ignoreFunctions.insert(std::string(FncsNamesToIgnore[i]));
  }
  return provideDefinitions(*unwrap(M), ignoreFunctions);
}
}

namespace {
class BCLoader final : public ModulePass {
public:
  static char ID;
  BCLoader() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return provideDefinitions(M, {}); }
};
} // namespace

char BCLoader::ID = 0;

static RegisterPass<BCLoader> X("bcloader",
                                "Link bitcode files for known functions");

ModulePass *createBCLoaderPass() { return new BCLoader(); }
