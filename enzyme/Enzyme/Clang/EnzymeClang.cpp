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
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;

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

void MakeGlobalOfFn(FunctionDecl *FD, CompilerInstance &CI) {
  // if (FD->isLateTemplateParsed()) return;
  // TODO save any type info into string like attribute
}

struct Visitor : public RecursiveASTVisitor<Visitor> {
  CompilerInstance &CI;
  Visitor(CompilerInstance &CI) : CI(CI) {}
  bool VisitFunctionDecl(FunctionDecl *FD) {
    MakeGlobalOfFn(FD, CI);
    return true;
  }
};

const char *enzyme_utils_file[] = {

};

class EnzymePlugin final : public clang::ASTConsumer {
  clang::CompilerInstance &CI;

public:
  EnzymePlugin(clang::CompilerInstance &CI) : CI(CI) {

    FrontendOptions &Opts = CI.getFrontendOpts();
    CodeGenOptions &CGOpts = CI.getCodeGenOpts();
    auto PluginName = "ClangEnzyme-" + std::to_string(LLVM_VERSION_MAJOR);
    for (auto P : Opts.Plugins)
      if (llvm::sys::path::stem(P).endswith(PluginName)) {
        bool contains = false;
        for (auto passPlugin : CGOpts.PassPlugins) {
          if (llvm::sys::path::stem(passPlugin).endswith(PluginName)) {
            contains = true;
            break;
          }
        }
        if (!contains) {
          CGOpts.PassPlugins.push_back(P);
          break;
        }
      }
    CI.getPreprocessorOpts().Includes.push_back("/enzyme/enzyme/version");

    std::string PredefineBuffer;
    PredefineBuffer.reserve(4080);
    llvm::raw_string_ostream Predefines(PredefineBuffer);
    Predefines << CI.getPreprocessor().getPredefines();
    MacroBuilder Builder(Predefines);
    Builder.defineMacro("ENZYME_VERSION_MAJOR",
                        std::to_string(ENZYME_VERSION_MAJOR));
    Builder.defineMacro("ENZYME_VERSION_MINOR",
                        std::to_string(ENZYME_VERSION_MINOR));
    Builder.defineMacro("ENZYME_VERSION_PATCH",
                        std::to_string(ENZYME_VERSION_PATCH));
    CI.getPreprocessor().setPredefines(Predefines.str());
  }
  ~EnzymePlugin() {}
  void HandleTranslationUnit(ASTContext &context) override {}
  bool HandleTopLevelDecl(clang::DeclGroupRef dg) override {
    using namespace clang;
    DeclGroupRef::iterator it;

    Visitor v(CI);
    // Forcibly require emission of all libdevice
    for (it = dg.begin(); it != dg.end(); ++it) {
      v.TraverseDecl(*it);
      if (auto FD = dyn_cast<FunctionDecl>(*it)) {
        if (!FD->hasAttr<clang::CUDADeviceAttr>())
          continue;

        if (!FD->getIdentifier())
          continue;
        if (!StringRef(FD->getLocation().printToString(CI.getSourceManager()))
                 .contains("/__clang_cuda_math.h"))
          continue;

        FD->addAttr(UsedAttr::CreateImplicit(CI.getASTContext()));
      }
      if (auto FD = dyn_cast<VarDecl>(*it)) {
        HandleCXXStaticMemberVarInstantiation(FD);
      }
    }
    return true;
  }
  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *V) override {
    if (!V->getIdentifier())
      return;
    auto name = V->getName();
    if (!(name.contains("__enzyme_inactive_global") ||
          name.contains("__enzyme_inactivefn") ||
          name.contains("__enzyme_function_like") ||
          name.contains("__enzyme_allocation_like") ||
          name.contains("__enzyme_register_gradient") ||
          name.contains("__enzyme_register_derivative") ||
          name.contains("__enzyme_register_splitderivative")))
      return;

    V->addAttr(clang::UsedAttr::CreateImplicit(CI.getASTContext()));
    return;
  }
};

// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<EnzymeAction<EnzymePlugin>>
    X("enzyme", "Enzyme Plugin");

namespace {

struct EnzymeFunctionLikeAttrInfo : public ParsedAttrInfo {
  EnzymeFunctionLikeAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
        {ParsedAttr::AS_GNU, "enzyme_function_like"},
        {ParsedAttr::AS_C2x, "enzyme_function_like"},
        {ParsedAttr::AS_CXX11, "enzyme_function_like"},
        {ParsedAttr::AS_CXX11, "enzyme::function_like"}};
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
    auto FD = cast<FunctionDecl>(D);
    if (Attr.getNumArgs() == 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_function_like' attribute requires a string argument");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    auto *Arg0 = Attr.getArgAsExpr(0);
    StringLiteral *Literal = dyn_cast<StringLiteral>(Arg0->IgnoreParenCasts());
    if (!Literal) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "first argument to the 'enzyme_function_like' "
          "attribute must be a string literal");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    // if (FD->isLateTemplateParsed()) return;
    auto &AST = S.getASTContext();
    DeclContext *declCtx = FD->getDeclContext();
    auto loc = FD->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, clang::TagTypeKind::TTK_Struct, declCtx,
                                 loc, loc, nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, clang::TagTypeKind::TTK_Struct, declCtx, loc,
                              loc, nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto Tinfo = nullptr;
    auto Tinfo0 = nullptr;
    auto FT = AST.getPointerType(FD->getType());
    auto CharTy = AST.getIntTypeForBitwidth(8, false);
    // VarDecl::Create(AST, RD, loc, loc, nullptr, FT, Tinfo, SC_None);
    auto FD0 = FieldDecl::Create(AST, RD, loc, loc, /*Ud*/ nullptr, FT, Tinfo0,
                                 /*expr*/ nullptr, /*mutable*/ true,
                                 /*inclassinit*/ ICIS_NoInit);
    FD0->setAccess(AS_public);
    RD->addDecl(FD0);
    auto FD1 = FieldDecl::Create(
        AST, RD, loc, loc, /*Ud*/ nullptr, AST.getPointerType(CharTy), Tinfo0,
        /*expr*/ nullptr, /*mutable*/ true, /*inclassinit*/ ICIS_NoInit);
    FD1->setAccess(AS_public);
    RD->addDecl(FD1);
    RD->completeDefinition();
    assert(RD->getDefinition());

    auto &Id = AST.Idents.get("__enzyme_function_like_autoreg_" +
                              FD->getNameAsString());
    auto T = AST.getRecordType(RD);
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, T, Tinfo, SC_None);
    // V->setStorageClass(SC_Extern);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), loc, FD, false,
                                  loc, FD->getType(), ExprValueKind::VK_LValue,
                                  FD, TemplateArgs);

    StringRef cstr = Literal->getString();
    Expr *exprs[2] = {
        ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                 DR, nullptr, ExprValueKind::VK_PRValue,
                                 FPOptionsOverride()),
        ImplicitCastExpr::Create(
            AST, AST.getPointerType(CharTy), CastKind::CK_ArrayToPointerDecay,
            StringLiteral::Create(
                AST, cstr, clang::StringLiteral::StringKind::Ordinary,
                /*Pascal*/ false,
                AST.getStringLiteralArrayType(CharTy, cstr.size()), loc),
            nullptr, ExprValueKind::VK_PRValue, FPOptionsOverride())};
    auto IL = new (AST) InitListExpr(AST, loc, exprs, loc);
    V->setInit(IL);
    IL->setType(T);
    if (IL->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "use of attribute 'enzyme_function_like' "
                                    "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
  }
};

} // namespace

static ParsedAttrInfoRegistry::Add<EnzymeFunctionLikeAttrInfo>
    X3("enzyme_function_like", "");
