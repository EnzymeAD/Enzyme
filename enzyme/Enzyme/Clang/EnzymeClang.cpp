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
#include "clang/Basic/FileManager.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

#include "../Utils.h"

#include "bundled_includes.h"

using namespace clang;

#if LLVM_VERSION_MAJOR >= 18
constexpr auto StructKind = clang::TagTypeKind::Struct;
#else
constexpr auto StructKind = clang::TagTypeKind::TTK_Struct;
#endif

#if LLVM_VERSION_MAJOR < 12
constexpr auto stringkind = clang::StringLiteral::StringKind::Ascii;
#endif

template <typename ConsumerType>
class EnzymeAction final : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new ConsumerType(CI));
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
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

#if LLVM_VERSION_MAJOR >= 18
void registerEnzyme(llvm::PassBuilder &PB);
#endif

class EnzymePlugin final : public clang::ASTConsumer {
  clang::CompilerInstance &CI;

public:
  EnzymePlugin(clang::CompilerInstance &CI) : CI(CI) {

    FrontendOptions &Opts = CI.getFrontendOpts();
    CodeGenOptions &CGOpts = CI.getCodeGenOpts();
    auto PluginName = "ClangEnzyme-" + std::to_string(LLVM_VERSION_MAJOR);
    bool contains = false;
#if LLVM_VERSION_MAJOR < 18
    std::string pluginPath;
#endif
    for (auto P : Opts.Plugins)
      if (endsWith(llvm::sys::path::stem(P), PluginName)) {
#if LLVM_VERSION_MAJOR < 18
        pluginPath = P;
#endif
        for (auto passPlugin : CGOpts.PassPlugins) {
          if (endsWith(llvm::sys::path::stem(passPlugin), PluginName)) {
            contains = true;
            break;
          }
        }
      }

    if (!contains) {
#if LLVM_VERSION_MAJOR >= 18
      CGOpts.PassBuilderCallbacks.push_back(registerEnzyme);
#else
      CGOpts.PassPlugins.push_back(pluginPath);
#endif
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

    auto baseFS = &CI.getFileManager().getVirtualFileSystem();
    llvm::vfs::OverlayFileSystem *fuseFS(
        new llvm::vfs::OverlayFileSystem(baseFS));
    IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs(
        new llvm::vfs::InMemoryFileSystem());

    struct tm y2k = {};

    y2k.tm_hour = 0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;
    time_t timer = mktime(&y2k);
    for (const auto &pair : include_headers) {
      fs->addFile(StringRef(pair[0]), timer,
                  llvm::MemoryBuffer::getMemBuffer(
                      StringRef(pair[1]), StringRef(pair[0]),
                      /*RequiresNullTerminator*/ true));
    }

    fuseFS->pushOverlay(fs);
    fuseFS->pushOverlay(baseFS);
    CI.getFileManager().setVirtualFileSystem(fuseFS);

    auto DE = CI.getFileManager().getDirectoryRef("/enzymeroot");
    assert(DE);
    auto DL = DirectoryLookup(*DE, SrcMgr::C_User,
                              /*isFramework=*/false);
    CI.getPreprocessor().getHeaderSearchInfo().AddSearchPath(DL,
                                                             /*isAngled=*/true);
  }
  ~EnzymePlugin() {}
  void HandleTranslationUnit(ASTContext &context) override {}
  bool HandleTopLevelDecl(clang::DeclGroupRef dg) override {
    using namespace clang;
    DeclGroupRef::iterator it;

    // Visitor v(CI);
    // Forcibly require emission of all libdevice
    for (it = dg.begin(); it != dg.end(); ++it) {
      // v.TraverseDecl(*it);
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
          name.contains("__enzyme_shouldrecompute") ||
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

#if LLVM_VERSION_MAJOR > 10
namespace {

struct EnzymeFunctionLikeAttrInfo : public ParsedAttrInfo {
  EnzymeFunctionLikeAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_function_like"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_function_like"},
#else
      {ParsedAttr::AS_C2x, "enzyme_function_like"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_function_like"},
      {ParsedAttr::AS_CXX11, "enzyme::function_like"}
    };
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
    if (Attr.getNumArgs() != 1) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_function' attribute requires a single string argument");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    auto *Arg0 = Attr.getArgAsExpr(0);
    StringLiteral *Literal = dyn_cast<StringLiteral>(Arg0->IgnoreParenCasts());
    if (!Literal) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "first argument to 'enzyme_function_like' "
                                    "attribute must be a string literal");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
#if LLVM_VERSION_MAJOR >= 12
    D->addAttr(AnnotateAttr::Create(
        S.Context, ("enzyme_function_like=" + Literal->getString()).str(),
        nullptr, 0, Attr.getRange()));
    return AttributeApplied;
#else
    auto FD = cast<FunctionDecl>(D);
    // if (FD->isLateTemplateParsed()) return;
    auto &AST = S.getASTContext();
    DeclContext *declCtx = FD->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = FD->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto Tinfo = nullptr;
    auto Tinfo0 = nullptr;
    auto FT = AST.getPointerType(FD->getType());
    auto CharTy = AST.getIntTypeForBitwidth(8, false);
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
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), loc, FD, false,
                                  loc, FD->getType(), ExprValueKind::VK_LValue,
                                  FD, TemplateArgs);
#if LLVM_VERSION_MAJOR >= 13
    auto rval = ExprValueKind::VK_PRValue;
#else
    auto rval = ExprValueKind::VK_RValue;
#endif
    StringRef cstr = Literal->getString();
    Expr *exprs[2] = {
#if LLVM_VERSION_MAJOR >= 12
      ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay, DR,
                               nullptr, rval, FPOptionsOverride()),
      ImplicitCastExpr::Create(
          AST, AST.getPointerType(CharTy), CastKind::CK_ArrayToPointerDecay,
          StringLiteral::Create(
              AST, cstr, stringkind,
              /*Pascal*/ false,
              AST.getStringLiteralArrayType(CharTy, cstr.size()), loc),
          nullptr, rval, FPOptionsOverride())
#else
      ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay, DR,
                               nullptr, rval),
      ImplicitCastExpr::Create(
          AST, AST.getPointerType(CharTy), CastKind::CK_ArrayToPointerDecay,
          StringLiteral::Create(
              AST, cstr, stringkind,
              /*Pascal*/ false,
              AST.getStringLiteralArrayType(CharTy, cstr.size()), loc),
          nullptr, rval)
#endif
    };
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
#endif
  }
};

#if LLVM_VERSION_MAJOR >= 12
static ParsedAttrInfoRegistry::Add<EnzymeFunctionLikeAttrInfo>
    X3("enzyme_function_like", "");

struct EnzymeShouldRecomputeAttrInfo : public ParsedAttrInfo {
  EnzymeShouldRecomputeAttrInfo() {
    OptArgs = 1;
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_shouldrecompute"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_shouldrecompute"},
#else
      {ParsedAttr::AS_C2x, "enzyme_shouldrecompute"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_shouldrecompute"},
      {ParsedAttr::AS_CXX11, "enzyme::shouldrecompute"}
    };
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    if (auto VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return true;
    }
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions and globals";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_inactive' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    D->addAttr(AnnotateAttr::Create(S.Context, "enzyme_shouldrecompute",
                                    nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeShouldRecomputeAttrInfo>
    ESR("enzyme_shouldrecompute", "");
#endif

struct EnzymeInactiveAttrInfo : public ParsedAttrInfo {
  EnzymeInactiveAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_inactive"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_inactive"},
#else
      {ParsedAttr::AS_C2x, "enzyme_inactive"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_inactive"},
      {ParsedAttr::AS_CXX11, "enzyme::inactive"}
    };
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    if (auto VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return true;
    }
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions and globals";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_inactive' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto &AST = S.getASTContext();
    DeclContext *declCtx = D->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = D->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto T = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getType()
                                  : cast<VarDecl>(D)->getType();
    auto Name = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getNameAsString()
                                     : cast<VarDecl>(D)->getNameAsString();
    auto FT = AST.getPointerType(T);
    auto subname = isa<FunctionDecl>(D) ? "inactivefn" : "inactive_global";
    auto &Id = AST.Idents.get(
        (StringRef("__enzyme_") + subname + "_autoreg_" + Name).str());
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, FT, nullptr, SC_None);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(D), false, loc, T,
        ExprValueKind::VK_LValue, cast<NamedDecl>(D), TemplateArgs);
#if LLVM_VERSION_MAJOR >= 13
    auto rval = ExprValueKind::VK_PRValue;
#else
    auto rval = ExprValueKind::VK_RValue;
#endif
    Expr *expr = nullptr;
    if (isa<FunctionDecl>(D)) {
#if LLVM_VERSION_MAJOR >= 12
      expr =
          ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                   DR, nullptr, rval, FPOptionsOverride());
#else
      expr = ImplicitCastExpr::Create(
          AST, FT, CastKind::CK_FunctionToPointerDecay, DR, nullptr, rval);
#endif
    } else {
      expr =
          UnaryOperator::Create(AST, DR, UnaryOperatorKind::UO_AddrOf, FT, rval,
                                clang::ExprObjectKind ::OK_Ordinary, loc,
                                /*canoverflow*/ false, FPOptionsOverride());
    }

    if (expr->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "use of attribute 'enzyme_inactive' "
                                    "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    V->setInit(expr);
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeInactiveAttrInfo> X4("enzyme_inactive",
                                                              "");

struct EnzymeNoFreeAttrInfo : public ParsedAttrInfo {
  EnzymeNoFreeAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_nofree"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_nofree"},
#else
      {ParsedAttr::AS_C2x, "enzyme_nofree"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_nofree"},
      {ParsedAttr::AS_CXX11, "enzyme::nofree"}
    };
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    if (auto VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return true;
    }
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions and globals";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_nofree' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto &AST = S.getASTContext();
    DeclContext *declCtx = D->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = D->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto T = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getType()
                                  : cast<VarDecl>(D)->getType();
    auto Name = isa<FunctionDecl>(D) ? cast<FunctionDecl>(D)->getNameAsString()
                                     : cast<VarDecl>(D)->getNameAsString();
    auto FT = AST.getPointerType(T);
    auto &Id = AST.Idents.get(
        (StringRef("__enzyme_nofree") + "_autoreg_" + Name).str());
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, FT, nullptr, SC_None);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(D), false, loc, T,
        ExprValueKind::VK_LValue, cast<NamedDecl>(D), TemplateArgs);
#if LLVM_VERSION_MAJOR >= 13
    auto rval = ExprValueKind::VK_PRValue;
#else
    auto rval = ExprValueKind::VK_RValue;
#endif
    Expr *expr = nullptr;
    if (isa<FunctionDecl>(D)) {
#if LLVM_VERSION_MAJOR >= 12
      expr =
          ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                   DR, nullptr, rval, FPOptionsOverride());
#else
      expr = ImplicitCastExpr::Create(
          AST, FT, CastKind::CK_FunctionToPointerDecay, DR, nullptr, rval);
#endif
    } else {
      expr =
          UnaryOperator::Create(AST, DR, UnaryOperatorKind::UO_AddrOf, FT, rval,
                                clang::ExprObjectKind ::OK_Ordinary, loc,
                                /*canoverflow*/ false, FPOptionsOverride());
    }

    if (expr->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error, "use of attribute 'enzyme_nofree' "
                                    "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    V->setInit(expr);
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeNoFreeAttrInfo> X5("enzyme_nofree",
                                                            "");

struct EnzymeSparseAccumulateAttrInfo : public ParsedAttrInfo {
  EnzymeSparseAccumulateAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_sparse_accumulate"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_sparse_accumulate"},
#else
      {ParsedAttr::AS_C2x, "enzyme_sparse_accumulate"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_sparse_accumulate"},
      {ParsedAttr::AS_CXX11, "enzyme::sparse_accumulate"}
    };
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    // This attribute appertains to functions only.
    if (isa<FunctionDecl>(D))
      return true;
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << Attr << "functions";
    return false;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    if (Attr.getNumArgs() != 0) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'enzyme_sparse_accumulate' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }

    auto &AST = S.getASTContext();
    DeclContext *declCtx = D->getDeclContext();
    for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
      if (tmpCtx->isRecord()) {
        declCtx = tmpCtx->getParent();
      }
    }
    auto loc = D->getLocation();
    RecordDecl *RD;
    if (S.getLangOpts().CPlusPlus)
      RD = CXXRecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                                 nullptr); // rId);
    else
      RD = RecordDecl::Create(AST, StructKind, declCtx, loc, loc,
                              nullptr); // rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto T = cast<FunctionDecl>(D)->getType();
    auto Name = cast<FunctionDecl>(D)->getNameAsString();
    auto FT = AST.getPointerType(T);
    auto &Id = AST.Idents.get(
        (StringRef("__enzyme_sparse_accumulate") + "_autoreg_" + Name).str());
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, FT, nullptr, SC_None);
    V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(AST));
    TemplateArgumentListInfo *TemplateArgs = nullptr;
    auto DR = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), loc, cast<ValueDecl>(D), false, loc, T,
        ExprValueKind::VK_LValue, cast<NamedDecl>(D), TemplateArgs);
#if LLVM_VERSION_MAJOR >= 13
    auto rval = ExprValueKind::VK_PRValue;
#else
    auto rval = ExprValueKind::VK_RValue;
#endif
    Expr *expr = nullptr;
#if LLVM_VERSION_MAJOR >= 12
    expr =
        ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                 DR, nullptr, rval, FPOptionsOverride());
#else
    expr = ImplicitCastExpr::Create(
        AST, FT, CastKind::CK_FunctionToPointerDecay, DR, nullptr, rval);
#endif

    if (expr->isValueDependent()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "use of attribute 'enzyme_sparse_accumulate' "
          "in a templated context not yet supported");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    V->setInit(expr);
    S.MarkVariableReferenced(loc, V);
    S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeSparseAccumulateAttrInfo>
    SparseX("enzyme_sparse_accumulate", "");
} // namespace

#endif
