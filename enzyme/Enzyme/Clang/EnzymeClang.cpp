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
#include "clang/Sema/Sema.h"
#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;

template <typename ConsumerType>
class EnzymeAction final : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) {
      llvm::errs() << " creating enzyme consumern\n";
    return std::unique_ptr<clang::ASTConsumer>(new ConsumerType(CI));
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) {
      llvm::errs() << " parsing ast enzyme action\n";
    return true;
  }

  PluginASTAction::ActionType getActionType() override {
    return AddBeforeMainAction;
  }
};

void MakeGlobalOfFn(FunctionDecl* FD, CompilerInstance &CI) {
    llvm::errs() << " FD: " << *FD << "\n";
    FD->dump();
    // if (FD->isLateTemplateParsed()) return;
    auto &AST = CI.getASTContext();
    DeclContext *declCtx = FD->getDeclContext();
    auto loc = FD->getLocation();
    RecordDecl *RD;
    if (CI.getLangOpts().CPlusPlus)
        RD = CXXRecordDecl::Create(AST, clang::TagTypeKind::TTK_Struct, declCtx, loc, loc, nullptr);//rId);
    else    
        RD = RecordDecl::Create(AST, clang::TagTypeKind::TTK_Struct, declCtx, loc, loc, nullptr);//rId);
    RD->setAnonymousStructOrUnion(true);
    RD->setImplicit();
    RD->startDefinition();
    auto Tinfo = nullptr;
    auto Tinfo0 = nullptr;
    auto FT = AST.getPointerType(FD->getType());
    auto CharTy = AST.getIntTypeForBitwidth(8, false);
    //VarDecl::Create(AST, RD, loc, loc, nullptr, FT, Tinfo, SC_None);
    auto FD0 = FieldDecl::Create(AST, RD, loc, loc, /*Ud*/nullptr, FT, Tinfo0, /*expr*/nullptr, /*mutable*/true, /*inclassinit*/ICIS_NoInit);
    FD0->setAccess(AS_public);
    RD->addDecl(FD0);
    auto FD1 = FieldDecl::Create(AST, RD, loc, loc, /*Ud*/nullptr, AST.getPointerType(CharTy), Tinfo0, /*expr*/nullptr, /*mutable*/true, /*inclassinit*/ICIS_NoInit);
    FD1->setAccess(AS_public);
    RD->addDecl(FD1);
    RD->completeDefinition();
    assert(RD->getDefinition());
    RD->dump();

    auto &Id = AST.Idents.get("__enzyme_global_name_thing_" + FD->getNameAsString());
    auto T= AST.getRecordType(RD);
    llvm::errs() << " T: "<< T << "\n";
    T->dump();
    auto V = VarDecl::Create(AST, declCtx, loc, loc, &Id, T, Tinfo, SC_None);
    V->setStorageClass(SC_Extern);
    // V->setStorageClass(SC_PrivateExtern);
    V->addAttr(clang::UsedAttr::CreateImplicit(CI.getASTContext()));
    auto &S = CI.getSema();
     TemplateArgumentListInfo *TemplateArgs=nullptr;
    auto DR = DeclRefExpr::Create(AST, NestedNameSpecifierLoc(), loc, FD, false, loc,  FD->getType(), ExprValueKind::VK_LValue, FD, TemplateArgs); 

    StringRef cstr = "tmp";
    Expr* exprs[2] = {
        ImplicitCastExpr::Create(AST, FT, CastKind::CK_FunctionToPointerDecay, DR, nullptr,  ExprValueKind::VK_LValue, FPOptionsOverride()),
        ImplicitCastExpr::Create(AST, FT, CastKind::CK_ArrayToPointerDecay, 
        StringLiteral::Create(AST, cstr, clang::StringLiteral::StringKind::Ordinary, /*Pascal*/false, AST.getStringLiteralArrayType(CharTy, cstr.size()), loc)
                , nullptr, ExprValueKind::VK_LValue, FPOptionsOverride())
    };
    auto IL = new (AST) InitListExpr(AST, loc, exprs, loc);
    V->setInit(IL);
    IL->setType(T);
    if (IL->isValueDependent()) return;
    V->dump();
    S.MarkVariableReferenced(loc, V);
    CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(V));
}

struct Visitor : public RecursiveASTVisitor<Visitor> {
    CompilerInstance &CI;
      Visitor(CompilerInstance &CI) : CI(CI) {}
      bool VisitFunctionDecl(FunctionDecl *FD) {
          MakeGlobalOfFn(FD, CI);
        return true;
      }

    };

class EnzymePlugin final : public clang::ASTConsumer {
  clang::CompilerInstance &CI;

public:
  EnzymePlugin(clang::CompilerInstance &CI) : CI(CI) {
      llvm::errs() << " creating enzyme plugin\n";
  }
  ~EnzymePlugin() {}
  void HandleTranslationUnit(ASTContext& context) override {
  }
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
