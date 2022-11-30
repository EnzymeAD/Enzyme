//===- EnzymeClang.cpp - Automatic Differentiation Transformation Pass ----===//
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
// This file contains a clang plugin for Enzyme.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

template <typename ConsumerType>
class EnzymeAction final : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) {
    return std::unique_ptr<clang::ASTConsumer>(new ConsumerType(CI));
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) {
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

class EnzymePlugin final : public clang::ASTConsumer {
  clang::CompilerInstance &CI;

public:
  EnzymePlugin(clang::CompilerInstance &CI) : CI(CI) {}
  ~EnzymePlugin() {}
  bool HandleTopLevelDecl(clang::DeclGroupRef dg) override {
    using namespace clang;
    DeclGroupRef::iterator it;

    // Forcibly require emission of all libdevice
    for (it = dg.begin(); it != dg.end(); ++it) {
      auto FD = dyn_cast<FunctionDecl>(*it);
      if (!FD)
        continue;

      if (!FD->hasAttr<clang::CUDADeviceAttr>())
        continue;

      if (!FD->getIdentifier())
        continue;
      if (!StringRef(FD->getLocation().printToString(CI.getSourceManager()))
               .contains("/__clang_cuda_math.h"))
        continue;

      FD->addAttr(UsedAttr::CreateImplicit(CI.getASTContext()));
    }
    return true;
  }
};

// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<EnzymeAction<EnzymePlugin>>
    X("enzyme", "Enzyme Plugin");


#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Attributes.h"
using namespace clang;

namespace {

struct EnzymeAttrInfo : public ParsedAttrInfo {
  EnzymeAttrInfo() {
    OptArgs = 2;
    // GNU-style __attribute__(("example")) and C++-style [[example]]
    static constexpr Spelling S[] = {{ParsedAttr::AS_GNU, "enzyme_allocator"},
                                     {ParsedAttr::AS_CXX11, "enzyme_allocator"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    // Check if the decl is at file scope.
    if (!D->getDeclContext()->isFileContext()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_allocator' attribute only allowed at file scope");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    if (Attr.getNumArgs() == 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_allocator' attribute requires argument of allocation size");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto *Arg0 = Attr.getArgAsExpr(0);
    IntegerLiteral *Literal =
        dyn_cast<IntegerLiteral>(Arg0->IgnoreParenCasts());
    if (!Literal) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "first argument to the 'enzyme_allocator' "
                                    "attribute must be a integer literal");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
      SmallVector<Expr *, 16> ArgsBuf;
      for (unsigned i = 0; i < Attr.getNumArgs(); i++) {
        ArgsBuf.push_back(Attr.getArgAsExpr(i));
      }
    D->addAttr(AnnotateAttr::Create(S.Context, "enzyme", ArgsBuf.data(),
                                    ArgsBuf.size(), Attr.getRange()));
    D->addAttr(Attribute::NoInline);
    return AttributeApplied;
  }
};

} // namespace

static ParsedAttrInfoRegistry::Add<EnzymeAttrInfo> X("enzyme", "Enzyme Plugin");

