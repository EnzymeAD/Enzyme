
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "opts.h"

using namespace llvm;

void emit_args(DagInit *dag, raw_ostream &os) {
  for (size_t i = 0; i < dag->getNumArgs(); ++i) {
    if (i > 0)
      os << ", ";
    if (auto def = dyn_cast<DefInit>(dag->getArg(i))) {
      auto Def = def->getDef();
      if (Def->isSubClassOf("Value")) {
        os << "tmp" << Def->getValueAsString("name");
      }
    } else {
      os << dag->getArgNameStr(i);
    }
  }
}
void emit_types(DagInit *dag, raw_ostream &os) {
  for (size_t i = 0; i < dag->getNumArgs(); ++i) {
    if (i > 0)
      os << ", ";
    if (auto def = dyn_cast<DefInit>(dag->getArg(i))) {
      auto Def = def->getDef();
      if (Def->isSubClassOf("Value")) {
        os << "tmp" << Def->getValueAsString("name") << "->getType()";
      }
    } else {
      os << dag->getArgNameStr(i) << "->getType()";
    }
  }
}

// class BlasOptPattern< list<dag> _inputs, list<string> _tmps, list<dag>
// _outputs> {
//   list<dag> inputs = _inputs;
//   // tmp variables will dissapear during the transformation
//   // and therefore are not allowed to be read elsewhere
//   list<string> tmps = _tmps;
//   list<dag> outputs = _outputs;
// }
// def first : BlasOptPattern<,
// [
// (b<"ger"> $layout, $m, $n, $alpha, $x, $incx, $y, $incy, $A, $lda),
// (b<"ger"> $layout, $m, $k, $beta, $v, $incv, $w, $incw, $B, $ldb),
// (b<"gemm"> $layout, $transa, $transb, $m, $n, $k, $alpha, $A, $lda, $B, $ldb,
// $beta, $C, $ldc),
// ],
// ["A", "B"],
// [
// (Value<1> (b<"dot"> $layout, $n, $v, $incv, $y, $incy)),
// (Value<2> (FMul $alpha Value<1>)),
// (Value<3> (FMul $beta Value<2>)),
// (b<"ger"> $layout, $m, $k, Value<3>, $x, $incx, $w, $incw, $C, $ldc),
// ]
// >;
void emitBlasOpt(StringRef name, std::vector<DagInit *> inputs,
                 std::vector<StringRef>, std::vector<DagInit *> outputs,
                 raw_ostream &os) {
  os << "bool opt" << name << "(llvm::Function *F, llvm::Module &M) {\n";
  os << "  using namespace llvm;\n";

  StringSet usedArgs{};
  std::vector<StringRef> functions{};
  StringMap<std::vector<DagInit *>> unique_functions{};
  for (auto input : inputs) {
    ArrayRef<StringInit *> args = input->getArgNames();
    auto Def = cast<DefInit>(input->getOperator())->getDef();
    assert(Def->isSubClassOf("b"));
    auto fnc_name = Def->getValueAsString("s");
    // auto fnc_name = input->getNameStr();
    functions.push_back(fnc_name);
    unique_functions[fnc_name].push_back(input);
    for (auto &arg : args) {
      if (usedArgs.count(arg->getValue()))
        continue;
      os << "  Value *" << arg->getValue() << " = nullptr;\n";
      usedArgs.insert(arg->getValue());
    }
  }

  os << "\n";

  for (auto fnc : unique_functions.keys()) {
    os << "  size_t idx_" << fnc << " = 0;\n";
  }

  os << "  // create a vector of calls to delete\n";
  os << "  std::vector<CallInst *> todelete;\n";
  os << "  int num_calls = 0;\n";

  os << "  for (auto &BB : *F) {\n"
     << "    for (auto &I : BB) {\n"
     << "      if (auto *CI = dyn_cast<CallInst>(&I)) {\n"
     << "        num_calls++;\n"
     << "        auto CIname = CI->getCalledFunction()->getName();\n"
     << "        auto blasOption = extractBLAS(CIname);\n"
     << "#if LLVM_VERSION_MAJOR >= 16\n"
     << "        if (!blasOption.has_value()) continue;\n"
     << "        auto blas = blasOption.value();\n"
     << "#else\n"
     << "        if (!blasOption.hasValue()) continue;\n"
     << "        auto blas = blasOption.getValue();\n"
     << "#endif\n";
  for (auto fnc : unique_functions.keys()) {
    os << "        if (blas.function == \"" << fnc << "\") {\n";
    std::string tab = "          ";
    auto fnc_vec = unique_functions[fnc];
    bool multiple = fnc_vec.size() > 1;
    os << tab << "assert(idx_" << fnc << " < " << fnc_vec.size()
       << " && \"idx out of bounds\");\n";
    os << tab << "std::vector<Value *> values;\n";
    for (size_t i = 0; i < fnc_vec.size(); ++i) {
      if (multiple) {
        os << tab << "if (idx_" << fnc << " == " << i << ")\n  ";
      }
      os << tab << "values = {";
      ArrayRef<StringInit *> args = fnc_vec[i]->getArgNames();
      bool first = true;
      for (auto arg : args) {
        os << (first ? "" : ", ") << arg->getValue();
        first = false;
      }
      os << "};\n";
    }
    os << tab << "bool set = cmp_or_set(CI, values);\n";
    os << tab << "if (!set) {\n";
    os << tab << "  llvm::errs() << \"args missmatch: " << fnc << "\";\n";
    os << tab << "  continue;\n";
    os << tab << "}\n";
    for (size_t i = 0; i < fnc_vec[0]->getNumArgs(); i++) {
      os << tab << "values[" << i << "] = CI->getArgOperand(" << i << ");\n";
    }
    for (size_t i = 0; i < fnc_vec.size(); ++i) {
      ArrayRef<StringInit *> args = fnc_vec[i]->getArgNames();
      size_t pos = 0;
      for (auto arg : args) {
        os << tab << arg->getValue() << " = CI->getArgOperand(" << i << ");\n";
        pos++;
      }
    }
    os << tab << "llvm::errs() << \"found " << fnc << "\\n\";\n";
    os << tab << "idx_" << fnc << "++;\n"
       << tab << "todelete.push_back(CI);\n"
       << tab << "continue;\n"
       << "        }\n";
  }
  os << "        llvm::errs() << \"unhandled: \" << blas.function << \"\\n\";\n";
  os << "      }\n";
  os << "    }\n";
  os << "  }\n";

  // check that all functions have been found
  os << "  bool found = true;\n";
  for (auto fnc : unique_functions.keys()) {
    os << "  if (idx_" << fnc << " != " << unique_functions[fnc].size() << ")\n"
       << "    found = false;\n";
  }
  os << "  if (!found) {\n"
     << "    llvm::errs() << \"num calls: \" << num_calls << \"\\n\";\n"
     << "    return false;\n"
     << "  }\n";

  os << "  llvm::errs() << \"found optimization " << name << "\\n\";\n";

  // now that we found an optimization to apply,
  // we can delete the old calls
  os << "  for (auto *CI : todelete) {\n"
     << "    CI->eraseFromParent();\n"
     << "  }\n";

  os << "  BasicBlock *bb = &F->getEntryBlock();\n"
     << "  IRBuilder<> B1(bb);\n";

  for (auto outerOutput : outputs) {
    DagInit *output = outerOutput;
    auto buffer = std::string("");
    auto Def = cast<DefInit>(output->getOperator())->getDef();
    if (Def->isSubClassOf("Value")) {
      assert(output->getNumArgs() == 1);
      auto name = Def->getValueAsString("name");
      buffer = (Twine("  Value *tmp") + name + " = ").str();
      // This is just wrapping the actual DagInit in a Value<>.
      // So now strip the Value wrapper to handle it in the next if/else
      output = cast<DagInit>(output->getArg(0));
      Def = cast<DefInit>(output->getOperator())->getDef();
    }
    if (Def->isSubClassOf("Inst")) {
      auto name = Def->getValueAsString("name");
      os << buffer << "  B1.Create" << name << "(";
      emit_args(output, os);
      os << ");\n";
    } else if (Def->isSubClassOf("b")) {
      auto fnc_name = Def->getValueAsString("s");
      if (unique_functions[fnc_name].size() >= 1) {
        // function decl already existed in the module
        os << "  Function *Fnc_" << fnc_name << " = M.getFunction(\""
           << fnc_name << "\");\n"
           << "  assert(Fnc_" << fnc_name << ");\n";
      } else {
        // if the function decl did not exist, we need to create it
        // iff the buffer is empty, we return void, othewise
        // the buffer would be equal to 'Value *tmp = ...'
        std::string retTy = "Type::getVoidTy(M.getContext())";
        if (!buffer.empty()) {
          if (fnc_name.contains("64")) {
            retTy = "Type::getDoubleTy(M.getContext())";
          } else {
            retTy = "Type::getFloatTy(M.getContext())";
          }
        }
        os << "  FunctionType *FT" << fnc_name << " = FunctionType::get("
           << retTy << ", {";
        emit_types(output, os);
        os << "}, false);\n";
        os << "  Function *Fnc_" << fnc_name
           << " = cast<Function>(M.getOrInsertFunction(\"" << fnc_name
           << "\", FT" << fnc_name << ").getCallee());\n";
      }
      os << buffer << "  B1.CreateCall(Fnc_" << fnc_name << ", {";
      emit_args(output, os);
      os << "});\n";
    } else {
      llvm::errs() << "failed with: " << Def->getName() << "\n";
      PrintFatalError(Def->getLoc(), "unknown output type");
      assert(false);
      llvm_unreachable("unknown output type");
    }
  }
  //// bb->setInsertPoint(insertionPoint);

  // B1.CreateRetVoid();

  os << "  return true;\n";
  os << "}\n";
}

void emitBlasOpts(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const char *patternNames = "BlasOptPattern";
  const auto &patterns = recordKeeper.getAllDerivedDefinitions(patternNames);

  for (Record *pattern : patterns) {
    ListInit *inputs = pattern->getValueAsListInit("inputs");
    std::vector<StringRef> tmps = pattern->getValueAsListOfStrings("tmps");
    ListInit *outputs = pattern->getValueAsListInit("outputs");

    std::vector<DagInit *> inputDags;
    for (auto input : *inputs) {
      DagInit *dag = dyn_cast<DagInit>(input);
      assert(dag);
      inputDags.push_back(dag);
    }
    std::vector<DagInit *> outputDags;
    for (auto output : *outputs) {
      DagInit *dag = dyn_cast<DagInit>(output);
      assert(dag);
      outputDags.push_back(dag);
    }
    emitBlasOpt(pattern->getName(), inputDags, tmps, outputDags, os);
  }
}
