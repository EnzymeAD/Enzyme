// An opt pass plugin for the tests of Model/, loaded after the Poseidon plugin,
// whose functions it calls.
//
// evaluate-fpcore parses each string of the module's !poseidon.test.fpcore
// named metadata as a Herbie expression without variables and prints it with
// its value from the MPFR reference (getMPFRValues at 53 bits, as the accuracy
// model takes it) and from the binary64 evaluation (getFPValues).
//
// lower-fpcore lowers each of them, as a rewrite is lowered, into a function
// @fpcore.<n>(double %x, double %y) of the module that returns it, its
// symbols x and y the two arguments.
#include "Support/Utils.h"
#include "Model/Evaluators.h"
#include "Model/FPCore.h"
#include "Model/FPNode.h"

#include "llvm/IR/IRBuilder.h"
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
      FPGraph G;
      FPNode *Node = cantFail(parseHerbieExpr(Expr, G));
      SmallVector<double, 1> Reference, Binary64;
      getMPFRValues({Node}, {}, Reference);
      getFPValues({Node}, {}, Binary64);
      outs() << Expr << ": reference " << format("%.17g", Reference[0])
             << " binary64 " << format("%.17g", Binary64[0]) << '\n';
    }
    return PreservedAnalyses::all();
  }
};

struct LowerFPCore : PassParent<LowerFPCore> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    NamedMDNode *Exprs = M.getNamedMetadata("poseidon.test.fpcore");
    if (!Exprs)
      report_fatal_error("lower-fpcore: no !poseidon.test.fpcore");
    LLVMContext &C = M.getContext();
    Type *Double = Type::getDoubleTy(C);
    for (auto [I, Entry] : enumerate(Exprs->operands())) {
      std::string Expr =
          cast<MDString>(Entry->getOperand(0))->getString().str();
      // Lowered into a scratch function first: the result's type, and so the
      // function's, is known only after lowering.
      Function *Scratch = Function::Create(
          FunctionType::get(Type::getVoidTy(C), {Double, Double}, false),
          GlobalValue::InternalLinkage, "", M);
      FPGraph G;
      for (auto [Symbol, Arg] : zip(StringRef("xy"), Scratch->args())) {
        auto *Node =
            G.create<FPLLValue>(&Arg, nullptr, &formatOf(FPKind::FP64));
        Node->symbol = std::string(1, Symbol);
        G.valueToNodeMap[&Arg] = Node;
        G.symbolToValueMap[Node->symbol] = &Arg;
      }
      FPNode *Node = cantFail(parseHerbieExpr(Expr, G));
      IRBuilder<> B(BasicBlock::Create(C, "entry", Scratch));
      Value *V = Node->getLLValue(B);
      B.CreateRet(V);
      Function *F = Function::Create(
          FunctionType::get(V->getType(), {Double, Double}, false),
          GlobalValue::ExternalLinkage, "fpcore." + Twine(I), M);
      for (auto [From, To, Name] :
           zip(Scratch->args(), F->args(), StringRef("xy"))) {
        To.setName(StringRef(&Name, 1));
        From.replaceAllUsesWith(&To);
      }
      F->splice(F->end(), Scratch);
      Scratch->eraseFromParent();
    }
    return PreservedAnalyses::none();
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
                  if (Name == "lower-fpcore") {
                    MPM.addPass(LowerFPCore());
                    return true;
                  }
                  return false;
                });
          }};
}
