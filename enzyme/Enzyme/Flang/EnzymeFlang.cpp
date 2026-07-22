//===- EnzymeFlang.cpp - Automatic Differentiation Transformation Pass ----===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains a *sketch* of a Flang frontend plugin for Enzyme.
//
// It mirrors the structure of the Clang plugin (Enzyme/Clang/EnzymeClang.cpp),
// but the two frontends expose very different plugin surfaces:
//
//   * The Clang plugin (a clang::PluginASTAction) runs *before* the main
//     compilation action and can inject the Enzyme LLVM pass directly into the
//     code-generation pipeline (CGOpts.PassBuilderCallbacks), so a single
//     `-fplugin=ClangEnzyme-XX` both wires up the pass and provides the
//     `enzyme_*` headers/attributes.
//
//   * Flang's plugin API (Fortran::frontend::PluginParseTreeAction) is instead
//     a *replacement* frontend action: it runs INSTEAD of code generation and
//     only has access to the (post-semantics) parse tree. It is also, per the
//     Flang driver documentation, limited to `flang -fc1` and currently only
//     available on Linux. There is no equivalent of Clang's
//     PassBuilderCallbacks hook, so this frontend plugin cannot itself inject
//     the Enzyme code-generation pass.
//
// Because this file is linked together with the rest of the Enzyme sources
// (${ENZYME_SRC}, which includes Enzyme.cpp's `llvmGetPassPluginInfo`), the
// resulting `FlangEnzyme-XX` shared object doubles as a standard LLVM pass
// plugin. That is how actual differentiation is performed today:
//
//     # 1. inspect Enzyme usage at the Flang frontend (this plugin):
//     flang -fc1 -load ./FlangEnzyme-XX.so -plugin enzyme input.f90
//
//     # 2. actually differentiate, via the LLVM pass plugin role of the *same*
//     #    shared object (see flang/docs/FlangDriver.md, `-fpass-plugin`):
//     flang -O2 -fpass-plugin=./FlangEnzyme-XX.so input.f90 -o input
//
// This sketch implements step 1: it walks the parse tree and reports every call
// to an Enzyme differentiation hook it finds. It is intended as a starting
// point that Enzyme maintainers can grow (see the TODO markers below).
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/parsing.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

namespace {

/// Returns true if `name` refers to one of the Enzyme differentiation hooks.
///
/// Fortran is case-insensitive, so Flang lowercases identifiers; we therefore
/// compare case-insensitively. Matching on the `enzyme_autodiff` /
/// `enzyme_fwddiff` substrings intentionally covers every spelling Enzyme
/// recognizes at the IR level: the `enzyme` module bindings
/// (`enzyme_autodiff`), the raw implicit-interface hooks (`f__enzyme_autodiff`),
/// and the C-style names (`__enzyme_autodiff`). See Enzyme/Fortran/enzyme.f90
/// and Enzyme/Fortran/enzyme_function_hooks.f90.
bool isEnzymeHook(llvm::StringRef name) {
  return name.contains_insensitive("enzyme_autodiff") ||
         name.contains_insensitive("enzyme_fwddiff");
}

/// Parse-tree visitor that reports calls to Enzyme differentiation hooks.
///
/// Both `CALL foo(...)` statements and function-style references `x = foo(...)`
/// route their callee through a `ProcedureDesignator`, so visiting that single
/// node type covers subroutine and function usage alike.
struct EnzymeParseTreeVisitor {
  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {}

  void Post(const Fortran::parser::ProcedureDesignator &pd) {
    const auto *name = std::get_if<Fortran::parser::Name>(&pd.u);
    if (!name)
      return;
    if (!isEnzymeHook(name->ToString()))
      return;

    ++count;
    llvm::outs() << "enzyme: detected Enzyme differentiation call to '"
                 << name->ToString() << "'\n";
  }

  unsigned count{0};
};

class EnzymeFlangAction final : public PluginParseTreeAction {
  // NOTE: the override is `executeAction` (lowercase e), per
  // flang/Frontend/FrontendActions.h. The `ExecuteAction` spelling that appears
  // in flang/docs/FlangDriver.md is stale.
  void executeAction() override {
    EnzymeParseTreeVisitor visitor;
    Fortran::parser::Walk(getParsing().parseTree(), visitor);

    llvm::outs() << "enzyme: " << visitor.count
                 << " Enzyme differentiation call(s) found\n";

    // TODO(enzyme): recognize Enzyme activity annotations at the frontend level,
    // analogous to the `enzyme_inactive` / `enzyme_function_like` / ...
    // ParsedAttrInfo registrations in Enzyme/Clang/EnzymeClang.cpp. Flang has no
    // custom-attribute plugin surface, so this would likely be driven off
    // directives/comments or naming conventions in the parse tree.

    // TODO(enzyme): once Flang exposes a code-generation / PassBuilder plugin
    // hook (it currently does not), register the Enzyme pass here so that a
    // single `-plugin enzyme` both inspects and differentiates. Until then,
    // differentiation is performed via the LLVM pass-plugin role of this shared
    // object using `-fpass-plugin=` (see the file header).
  }
};

} // namespace

// Register the plugin so it can be selected with `-plugin enzyme`.
static FrontendPluginRegistry::Add<EnzymeFlangAction>
    X("enzyme", "Enzyme automatic differentiation plugin (sketch)");
