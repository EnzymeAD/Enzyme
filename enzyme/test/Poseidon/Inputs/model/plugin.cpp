// An opt pass plugin for the tests of Model/, loaded after the Poseidon plugin,
// whose functions it calls.
//
// evaluate-fpcore parses each string of the module's !poseidon.test.fpcore
// named metadata as a Herbie expression without variables and prints it with
// its value from the MPFR reference (getMPFRValues at 53 bits, as the accuracy
// model takes it) and from the binary64 evaluation (getFPValues).
#include "IR/Utils.h"
#include "Model/Evaluators.h"
#include "Model/FPCore.h"
#include "Model/FPNode.h"

#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#if LLVM_VERSION_MAJOR >= 22 && !defined(ENZYME_LLVM22_PRERELEASE)
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace poseidon;

namespace {

struct EvaluateFPCore : PassParent<EvaluateFPCore> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    NamedMDNode *Exprs = M.getNamedMetadata("poseidon.test.fpcore");
    if (!Exprs)
      report_fatal_error("evaluate-fpcore: no !poseidon.test.fpcore");
    for (MDNode *Entry : Exprs->operands()) {
      std::string Expr =
          cast<MDString>(Entry->getOperand(0))->getString().str();
      std::unordered_map<Value *, std::shared_ptr<FPNode>> ValueToNode;
      std::unordered_map<std::string, Value *> SymbolToValue;
      std::shared_ptr<FPNode> Node =
          parseHerbieExpr(Expr, ValueToNode, SymbolToValue);
      SmallVector<double, 1> Reference, Binary64;
      getMPFRValues({Node.get()}, {}, Reference, 53);
      getFPValues({Node.get()}, {}, Binary64);
      outs() << Expr << ": reference " << format("%.17g", Reference[0])
             << " binary64 " << format("%.17g", Binary64[0]) << '\n';
    }
    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "PoseidonTestModel", "0",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "evaluate-fpcore") {
                    MPM.addPass(EvaluateFPCore());
                    return true;
                  }
                  return false;
                });
          }};
}
