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
extern "C" void registerEnzyme(llvm::PassBuilder &PB);
#endif

namespace {

/// Whether this declaration of D is a definition, and hence whether clang will
/// emit an annotation on it -- annotations are only emitted for the entities a
/// translation unit defines.
bool isDefinitionInThisTU(const Decl *D) {
  if (auto FD = dyn_cast<FunctionDecl>(D))
    return FD->isThisDeclarationADefinition();
  if (auto VD = dyn_cast<VarDecl>(D))
    return VD->isThisDeclarationADefinition() != VarDecl::DeclarationOnly;
  return false;
}

/// Synthesize the global registering D with Enzyme, whose initializer is the
/// address of D. Kind is the Enzyme registration kind, e.g. "inactivefn".
/// Returns false if D is templated, and so has no address to register yet.
bool registerEnzymeGlobal(Sema &S, Decl *D, StringRef Kind) {
  auto &AST = S.getASTContext();

  // The registration global is emitted alongside D, so step out of any
  // enclosing record.
  DeclContext *declCtx = D->getDeclContext();
  for (auto tmpCtx = declCtx; tmpCtx; tmpCtx = tmpCtx->getParent()) {
    if (tmpCtx->isRecord()) {
      declCtx = tmpCtx->getParent();
    }
  }

  auto VD = cast<ValueDecl>(D);
  auto loc = D->getLocation();
  auto T = VD->getType();
  auto FT = AST.getPointerType(T);

  auto DR = DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), loc, VD, false,
                                loc, T, ExprValueKind::VK_LValue, VD,
                                /*TemplateArgs*/ nullptr);
  auto rval = ExprValueKind::VK_PRValue;
  Expr *expr = nullptr;
  if (isa<FunctionDecl>(D)) {
    expr =
        ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay,
                                 DR, nullptr, rval, FPOptionsOverride());
  } else {
    expr = UnaryOperator::Create(AST, DR, UnaryOperatorKind::UO_AddrOf, FT,
                                 rval, clang::ExprObjectKind ::OK_Ordinary, loc,
                                 /*canoverflow*/ false, FPOptionsOverride());
  }

  // A templated declaration has no address until it is instantiated.
  if (expr->isValueDependent())
    return false;

  auto &Id = AST.Idents.get(
      (StringRef("__enzyme_") + Kind + "_autoreg_" + VD->getNameAsString())
          .str());
  auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, FT, nullptr, SC_None);
  V->setStorageClass(SC_PrivateExtern);
  V->addAttr(clang::UsedAttr::CreateImplicit(AST));
  V->setInit(expr);
  S.MarkVariableReferenced(loc, V);
  S.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
  return true;
}

/// The Enzyme registration kind matching an annotation emitted by one of the
/// marker attributes, or the empty string for any other annotation.
StringRef enzymeRegistrationKind(StringRef Annotation, const Decl *D) {
  if (isa<FunctionDecl>(D))
    return llvm::StringSwitch<StringRef>(Annotation)
        .Case("enzyme_inactivefn", "inactivefn")
        .Case("enzyme_inactivenoblockfn", "inactivenoblockfn")
        .Case("enzyme_nofree", "nofree")
        .Case("enzyme_sparse_accumulate", "sparse_accumulate")
        .Default("");
  if (isa<VarDecl>(D))
    return llvm::StringSwitch<StringRef>(Annotation)
        .Case("enzyme_inactive", "inactive_global")
        .Case("enzyme_nofree", "nofree")
        .Default("");
  return "";
}

/// Clang only emits annotations for the entities a translation unit defines, so
/// a declaration carrying one of the marker annotations but defined elsewhere
/// needs a registration global to force the reference instead.
void registerEnzymeDeclIfNotDefinedHere(Sema &S, Decl *D) {
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D))
    return;
  if (isDefinitionInThisTU(D))
    return;
  for (auto A : D->specific_attrs<AnnotateAttr>()) {
    StringRef Kind = enzymeRegistrationKind(A->getAnnotation(), D);
    if (!Kind.empty())
      registerEnzymeGlobal(S, D, Kind);
  }
}

} // namespace

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
      // Whether a declaration is defined here is only settled once it reaches
      // the consumer, so this is where a marked declaration defined in another
      // translation unit picks up its registration global.
      registerEnzymeDeclIfNotDefinedHere(CI.getSema(), *it);
      if (auto FD = dyn_cast<FunctionDecl>(*it)) {
        if (!FD->hasAttr<clang::CUDADeviceAttr>())
          continue;

        if (!FD->getIdentifier())
          continue;
        if (!StringRef(FD->getLocation().printToString(CI.getSourceManager()))
                 .contains("/__clang_cuda_math.h") &&
            !StringRef(FD->getLocation().printToString(CI.getSourceManager()))
                 .contains("/__clang_hip_math.h"))
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
          name.contains("__enzyme_inactivenoblockfn") ||
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

bool isGlobalDecl(const Decl *D) {
  auto VD = dyn_cast<VarDecl>(D);
  return VD && VD->hasGlobalStorage();
}

/// Shared check for the registration attributes which apply to both functions
/// and global variables.
bool appertainsToFunctionOrGlobal(Sema &S, const ParsedAttr &Attr,
                                  const Decl *D) {
  if (isa<FunctionDecl>(D) || isGlobalDecl(D))
    return true;

  S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
      << Attr << " applies to functions and globals only";
  return false;
}

/// Shared implementation of the attributes which mark a declaration for Enzyme.
///
/// These lower to an annotation, which clang propagates through template
/// instantiation for us, so the attribute also applies to declarations written
/// inside a template -- where the alternative of taking the address of the
/// declaration is not possible until instantiation. FnAnnotation is used for
/// functions and VarAnnotation for variables.
///
/// Clang only emits annotations for the entities a translation unit defines,
/// so a declaration defined elsewhere additionally needs a registration global
/// to force the reference. Whether a function declarator will have a body is
/// not yet known here, as attributes are processed before the body is parsed,
/// so that is left to registerEnzymeDeclIfNotDefinedHere once the declaration
/// reaches the AST consumer.
///
/// A class member never reaches the consumer on its own, so it is registered
/// here instead, and for the same reason without knowing whether it is defined
/// in this translation unit. A member defined here therefore gets both
/// lowerings; they mark it identically, and the redundant global is consumed by
/// PreserveNVVM like any other.
ParsedAttrInfo::AttrHandling handleEnzymeMarkerAttr(Sema &S, Decl *D,
                                                    const ParsedAttr &Attr,
                                                    StringRef AttrName,
                                                    StringRef FnAnnotation,
                                                    StringRef VarAnnotation) {
  if (Attr.getNumArgs() != 0) {
    unsigned ID = S.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Error, "'%0' attribute requires zero arguments");
    S.Diag(Attr.getLoc(), ID) << AttrName;
    return ParsedAttrInfo::AttributeNotApplied;
  }

  StringRef Annotation = isa<FunctionDecl>(D) ? FnAnnotation : VarAnnotation;
  D->addAttr(
      AnnotateAttr::Create(S.Context, Annotation, nullptr, 0, Attr.getRange()));

  if (D->getDeclContext()->isRecord())
    registerEnzymeDeclIfNotDefinedHere(S, D);
  return ParsedAttrInfo::AttributeApplied;
}

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
    D->addAttr(AnnotateAttr::Create(
        S.Context, ("enzyme_function_like=" + Literal->getString()).str(),
        nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};

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
    return appertainsToFunctionOrGlobal(S, Attr, D);
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    return handleEnzymeMarkerAttr(S, D, Attr, "enzyme_inactive",
                                  /*FnAnnotation*/ "enzyme_inactivefn",
                                  /*VarAnnotation*/ "enzyme_inactive");
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeInactiveAttrInfo> X4("enzyme_inactive",
                                                              "");

struct EnzymeInactiveNoblockAttrInfo : public ParsedAttrInfo {
  EnzymeInactiveNoblockAttrInfo() {
    OptArgs = 1;
    // GNU-style __attribute__(("example")) and C++/C2x-style [[example]] and
    // [[plugin::example]] supported.
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_inactive_noblock"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_inactive_noblock"},
#else
      {ParsedAttr::AS_C2x, "enzyme_inactive_noblock"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_inactive_noblock"},
      {ParsedAttr::AS_CXX11, "enzyme::inactive_noblock"}
    };
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    return appertainsToFunctionOrGlobal(S, Attr, D);
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    return handleEnzymeMarkerAttr(S, D, Attr, "enzyme_inactive_noblock",
                                  /*FnAnnotation*/ "enzyme_inactivenoblockfn",
                                  /*VarAnnotation*/ "enzyme_inactive");
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeInactiveNoblockAttrInfo>
    X4_nb("enzyme_inactive_noblock", "");

struct EnzymeElementwiseReadAttrInfo : public ParsedAttrInfo {
  EnzymeElementwiseReadAttrInfo() {
    OptArgs = 1;
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_elementwise_read"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_elementwise_read"},
#else
      {ParsedAttr::AS_C2x, "enzyme_elementwise_read"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_elementwise_read"},
      {ParsedAttr::AS_CXX11, "enzyme::elementwise_read"}
    };
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
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
          "'enzyme_elementwise_read' attribute requires zero arguments");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    D->addAttr(AnnotateAttr::Create(S.Context, "enzyme_elementwise_read",
                                    nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeElementwiseReadAttrInfo>
    XElemRead("enzyme_elementwise_read", "");

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
    return appertainsToFunctionOrGlobal(S, Attr, D);
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    return handleEnzymeMarkerAttr(S, D, Attr, "enzyme_nofree",
                                  /*FnAnnotation*/ "enzyme_nofree",
                                  /*VarAnnotation*/ "enzyme_nofree");
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
    return handleEnzymeMarkerAttr(S, D, Attr, "enzyme_sparse_accumulate",
                                  /*FnAnnotation*/ "enzyme_sparse_accumulate",
                                  /*VarAnnotation*/ "enzyme_sparse_accumulate");
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeSparseAccumulateAttrInfo>
    SparseX("enzyme_sparse_accumulate", "");

struct EnzymeNoTypeAnalysisAttrInfo : public ParsedAttrInfo {
  EnzymeNoTypeAnalysisAttrInfo() {
    static constexpr Spelling S[] = {
      {ParsedAttr::AS_GNU, "enzyme_notypeanalysis"},
#if LLVM_VERSION_MAJOR > 17
      {ParsedAttr::AS_C23, "enzyme_notypeanalysis"},
#else
      {ParsedAttr::AS_C2x, "enzyme_notypeanalysis"},
#endif
      {ParsedAttr::AS_CXX11, "enzyme_notypeanalysis"},
      {ParsedAttr::AS_CXX11, "enzyme::notypeanalysis"}
    };
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr,
                            const Decl *D) const override {
    return appertainsToFunctionOrGlobal(S, Attr, D);
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D,
                                   const ParsedAttr &Attr) const override {
    // For now enzyme::notypeanalysis corresponds to the internal attribute enzyme_ta_norecur
    return handleEnzymeMarkerAttr(S, D, Attr, "enzyme_notypeanalysis",
                                    /*FnAnnotation*/ "enzyme_ta_norecur",
                                    /*VarAnnotation*/ "enzyme_ta_norecur");
  }
};

static ParsedAttrInfoRegistry::Add<EnzymeNoTypeAnalysisAttrInfo> enzyme_notypeanalysis("enzyme_notypeanalysis", "");

} // namespace

#endif
