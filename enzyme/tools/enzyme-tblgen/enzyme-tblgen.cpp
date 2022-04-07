//===- enzyme-tblgen.cpp - Top-Level TableGen implementation for Enzyme -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for Enzyme's TableGen.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

enum ActionType { GenRewriters };

static cl::opt<ActionType>
    action(cl::desc("Action to perform:"),
           cl::values(clEnumValN(GenRewriters, "gen-rewriters",
                                 "Generate rewriter definitions")));

static void emitRewriters(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = recordKeeper.getAllDerivedDefinitions("Pattern");
  Record *attrClass = recordKeeper.getClass("Attr");

  // Ensure unique patterns simply by appending unique suffix.
  unsigned rewritePatternCount = 0;
  std::string baseRewriteName = "GeneratedConvert";
  for (Record *pattern : patterns) {
    DagInit *tree = pattern->getValueAsDag("PatternToMatch");

    StringMap<int> nameToOrdinal;
    for (int i = 0, e = tree->getNumArgs(); i != e; ++i)
      nameToOrdinal[tree->getArgNameStr(i)] = i;

    // TODO(jpienaar): Expand to multiple matches.
    for (auto arg : tree->getArgs()) {
      if (isa<DagInit>(arg))
        PrintFatalError(pattern->getLoc(),
                        "only single pattern inputs supported");
    }

    // Emit RewritePattern for Pattern.
    DefInit *root = cast<DefInit>(tree->getOperator());
    std::string rewriteName =
        baseRewriteName + llvm::utostr(rewritePatternCount++);
    auto *rootName = cast<StringInit>(root->getDef()->getValueInit("opName"));
    os << "struct " << rewriteName << " : public RewritePattern {\n"
       << "  " << rewriteName << "(MLIRContext *context) : RewritePattern("
       << rootName->getAsString() << ", 1, context) {}\n"
       << "  PatternMatchResult match(Operation *op) const override {\n"
       << "    // TODO: This just handle 1 result\n"
       << "    if (op->getNumResults() != 1) return matchFailure();\n"
       << "    return matchSuccess();\n  }\n";

    ListInit *resultOps = pattern->getValueAsListInit("ResultOps");
    if (resultOps->size() != 1)
      PrintFatalError("only single result rules supported");
    DagInit *resultTree = cast<DagInit>(resultOps->getElement(0));

    // TODO(jpienaar): Expand to multiple results.
    for (auto result : resultTree->getArgs()) {
      if (isa<DagInit>(result))
        PrintFatalError(pattern->getLoc(), "only single op result supported");
    }
    DefInit *resultRoot = cast<DefInit>(resultTree->getOperator());
    std::string opName = resultRoot->getAsUnquotedString();
    auto resultOperands = resultRoot->getDef()->getValueAsDag("arguments");

    SmallVector<StringRef, 2> split;
    SplitString(opName, split, "_");
    auto className = join(split, "::");
    os << formatv(R"(
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto* context = op->getContext(); (void)context;
    rewriter.replaceOpWithNewOp<{0}>(op, op->getResult(0)->getType())",
                  className);
    if (resultOperands->getNumArgs() != resultTree->getNumArgs()) {
      PrintFatalError(pattern->getLoc(),
                      Twine("mismatch between arguments of resultant op (") +
                          Twine(resultOperands->getNumArgs()) +
                          ") and arguments provided for rewrite (" +
                          Twine(resultTree->getNumArgs()) + Twine(')'));
    }

    // Create the builder call for the result.
    for (int i = 0, e = resultTree->getNumArgs(); i != e; ++i) {
      // Start each operand on its own line.
      (os << ",\n").indent(6);

      auto *arg = resultTree->getArg(i);
      std::string name = resultTree->getArgName(i)->getAsUnquotedString();
      auto defInit = dyn_cast<DefInit>(arg);

      auto *argument = resultOperands->getArg(i);
      auto argumentDefInit = dyn_cast<DefInit>(argument);
      bool argumentIsAttr = false;
      if (argumentDefInit) {
        if (auto recTy = dyn_cast<RecordRecTy>(argumentDefInit->getType()))
          argumentIsAttr = recTy->isSubClassOf(attrClass);
      }

      if (argumentIsAttr) {
        if (!defInit) {
          std::string argumentName =
              resultOperands->getArgName(i)->getAsUnquotedString();
          PrintFatalError(pattern->getLoc(),
                          Twine("attribute '") + argumentName +
                              "' needs to be constant initialized");
        }

        auto value = defInit->getDef()->getValue("value");
        if (!value)
          PrintFatalError(pattern->getLoc(), Twine("'value' not defined in ") +
                                                 arg->getAsString());

        switch (value->getType()->getRecTyKind()) {
        case RecTy::IntRecTyKind:
          // TODO(jpienaar): This is using 64-bits for all the bitwidth of the
          // type could instead be queried. These are expected to be mostly used
          // for enums or constant indices and so no arithmetic operations are
          // expected on these.
          os << formatv(
              "/*{0}=*/IntegerAttr::get(Type::getInteger(64, context), {1})",
              name, value->getValue()->getAsString());
          break;
        case RecTy::StringRecTyKind:
          os << formatv("/*{0}=*/StringAttr::get({1}, context)", name,
                        value->getValue()->getAsString());
          break;
        default:
          PrintFatalError(pattern->getLoc(),
                          Twine("unsupported/unimplemented value type for ") +
                              name);
        }
        continue;
      }

      // Verify the types match between the rewriter's result and the
      if (defInit && argumentDefInit &&
          defInit->getType() != argumentDefInit->getType()) {
        PrintFatalError(
            pattern->getLoc(),
            "mismatch in type of operation's argument and rewrite argument " +
                Twine(i));
      }

      // Lookup the ordinal for the named operand.
      auto ord = nameToOrdinal.find(name);
      if (ord == nameToOrdinal.end())
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand '") + name + "'");
      os << "/*" << name << "=*/op->getOperand(" << ord->getValue() << ")";
    }
    os << "\n    );\n  }\n};\n";
  }

  // Emit function to add the generated matchers to the pattern list.
  os << "void populateWithGenerated(MLIRContext *context, "
     << "OwningRewritePatternList *patterns) {\n";
  for (unsigned i = 0; i != rewritePatternCount; ++i) {
    os << " patterns->push_back(std::make_unique<" << baseRewriteName << i
       << ">(context));\n";
  }
  os << "}\n";
}

static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case GenRewriters:
    emitRewriters(records, os);
    return false;
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &EnzymeTableGenMain);
}