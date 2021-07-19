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

cl::opt<std::string> BCPath("bcpath", cl::init(""), cl::Hidden,
                            cl::desc("Path to BC definitions"));

namespace {
class BCLoader : public ModulePass {
public:
  static char ID;
  BCLoader() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    std::map<std::string, std::string> bcmap = {
        {"cblas_ddot", "cblas_ddot_double.bc"},
        {"cblas_daxpy", "cblas_daxpy_void.bc"}};
    std::set<std::string> funcnames;
    for (auto &it : bcmap) {
      funcnames.insert(it.first);
    }
    std::set<std::string> funcspresent;
    for (Function &f : M) {
      if (funcnames.find(f.getName().str()) != funcnames.end())
        funcspresent.insert(f.getName().str());
    }
    for (auto &f : funcspresent) {
      SMDiagnostic Err;
#if LLVM_VERSION_MAJOR <= 10
      auto BC = llvm::parseIRFile(BCPath + "/" + bcmap.find(f)->second, Err,
                                  M.getContext(), true,
                                  M.getDataLayout().getStringRepresentation());
#else
      auto BC =
          llvm::parseIRFile(BCPath + "/" + bcmap.find(f)->second, Err,
                            M.getContext(), [&](StringRef) {
                              return Optional<std::string>(
                                  M.getDataLayout().getStringRepresentation());
                            });
#endif
        if (!BC)
          Err.print("bcloader", llvm::errs());
        assert(BC);
        Linker L(M);
        L.linkInModule(std::move(BC));
    }
    return true;
  }
};
} // namespace

char BCLoader::ID = 0;

static RegisterPass<BCLoader> X("bcloader",
                                "Link bitcode files for known functions");

ModulePass *createBCLoaderPass() { return new BCLoader(); }
