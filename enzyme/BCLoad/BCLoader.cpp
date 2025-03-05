#include "llvm/IR/IRBuilder.h"
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

static inline bool endsWith(llvm::StringRef string, llvm::StringRef suffix) {
#if LLVM_VERSION_MAJOR >= 18
  return string.ends_with(suffix);
#else
  return string.endswith(suffix);
#endif // LLVM_VERSION_MAJOR
}

static inline llvm::StringRef getFuncName(llvm::Function *called) {
  if (called->hasFnAttribute("enzyme_math"))
    return called->getFnAttribute("enzyme_math").getValueAsString();
  else if (called->hasFnAttribute("enzyme_allocator"))
    return "enzyme_allocator";
  else
    return called->getName();
}

bool provideDefinitions(Module &M, std::set<std::string> ignoreFunctions,
                        std::vector<std::string> &replaced) {
  std::vector<StringRef> todo;
  bool seen32 = false;
  bool seen64 = false;
  std::vector<std::pair<StringRef, llvm::Function *>> name_rewrites;
  for (auto &F : M) {
    if (!F.empty())
      continue;
    auto name = getFuncName(&F);
    if (ignoreFunctions.count(name.str()))
      continue;
    int index = 0;
    for (auto postfix : {"", "_", "_64_"}) {
      std::string str;
      if (strlen(postfix) == 0) {
        str = name.str();
      } else if (endsWith(name, postfix)) {
        auto blasName = name.substr(0, name.size() - strlen(postfix)).str();
        str = "cblas_" + blasName;
      }

      auto found = EnzymeBlasBC.find(str);
      if (found != EnzymeBlasBC.end()) {
        replaced.push_back(name.str());
        todo.push_back(found->second);
        if (name != F.getName()) {
          name_rewrites.emplace_back(name, &F);
        }
        if (index == 1)
          seen32 = true;
        if (index == 2)
          seen64 = true;
        break;
      }
      index++;
    }
  }

  for (auto &&[realname, F] : name_rewrites) {
    auto decl = M.getOrInsertFunction(realname, F->getFunctionType());
    auto entry = BasicBlock::Create(F->getContext(), "entry", F);
    IRBuilder<> B(entry);
    SmallVector<Value *, 1> vals;
    for (auto &arg : F->args())
      vals.push_back(&arg);
    auto rt = B.CreateCall(decl, vals);
    if (rt->getType()->isVoidTy())
      B.CreateRetVoid();
    else
      B.CreateRet(rt);
  }

  // Push fortran wrapper libs before all the other blas
  // to ensure the fortran injections have their code
  // replaced
  if (seen32)
    todo.insert(todo.begin(), __data_fblas32);
  if (seen64)
    todo.insert(todo.begin(), __data_fblas64);
  bool changed = false;
  for (auto mod : todo) {
    SMDiagnostic Err;
    MemoryBufferRef buf(mod, StringRef("bcloader"));

#if LLVM_VERSION_MAJOR >= 16
    auto BC = llvm::parseIR(buf, Err, M.getContext(),
                            llvm::ParserCallbacks([&](StringRef, StringRef) {
                              return std::optional<std::string>(
                                  M.getDataLayout().getStringRepresentation());
                            }));
#else
    auto BC = llvm::parseIR(buf, Err, M.getContext(), [&](StringRef) {
      return Optional<std::string>(M.getDataLayout().getStringRepresentation());
    });
#endif

    if (!BC) {
      Err.print("bcloader", llvm::errs());
      continue;
    }
    assert(BC);
    SmallVector<std::string, 1> toReplace;
    for (auto &F : *BC) {
      if (F.empty())
        continue;
      auto name = getFuncName(&F);
      if (ignoreFunctions.count(name.str())) {
        F.dropAllReferences();
#if LLVM_VERSION_MAJOR >= 16
        F.erase(F.begin(), F.end());
#else
        F.getBasicBlockList().erase(F.begin(), F.end());
#endif
        continue;
      }
      toReplace.push_back(name.str());
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
                                 size_t numFncNames, const char ***foundP,
                                 size_t *foundLen) {
  std::set<std::string> ignoreFunctions = {};
  for (size_t i = 0; i < numFncNames; i++) {
    ignoreFunctions.insert(std::string(FncsNamesToIgnore[i]));
  }
  std::vector<std::string> replaced;
  auto res = provideDefinitions(*unwrap(M), ignoreFunctions, replaced);

  const char **found = nullptr;
  if (replaced.size()) {
    found = (const char **)malloc(replaced.size() * sizeof(const char **));
    for (size_t i = 0; i < replaced.size(); i++) {
      char *data = (char *)malloc(replaced[i].size() + 1);
      memcpy(data, replaced[i].data(), replaced[i].size());
      data[replaced[i].size()] = 0;
      found[i] = data;
    }
  }
  *foundP = found;
  *foundLen = replaced.size();

  return res;
}
}

namespace {
class BCLoader final : public ModulePass {
public:
  static char ID;
  BCLoader() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    std::vector<std::string> replaced;
    return provideDefinitions(M, {}, replaced);
  }
};
} // namespace

char BCLoader::ID = 0;

static RegisterPass<BCLoader> X("bcloader",
                                "Link bitcode files for known functions");

ModulePass *createBCLoaderPass() { return new BCLoader(); }
