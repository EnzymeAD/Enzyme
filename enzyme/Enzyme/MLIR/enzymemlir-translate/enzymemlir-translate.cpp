//===- enzymemlir-translate.cpp - The enzymemlir-translate driver ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'enzymemlir-translate' tool, which is the enzyme
// analog of mlir-translate, used to drive lowering to LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

using namespace mlir;
using namespace llvm;
class ActivityToMetadataTranslation : public LLVMTranslationDialectInterface {
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  void annotateActivity(Operation *op, StringRef key,
                        LLVM::ModuleTranslation &moduleTranslation) const {
    LLVMContext &llvmCtx = moduleTranslation.getLLVMContext();
    MDNode *md = MDNode::get(llvmCtx, {});

    if (op->getNumResults() == 1) {
      llvm::Value *val = moduleTranslation.lookupValue(op->getResult(0));
      if (auto *inst = dyn_cast<llvm::Instruction>(val)) {
        inst->setMetadata(key, md);
      }
    } else if (op->getNumResults() == 0) {
      llvm::Instruction *inst = moduleTranslation.lookupOperation(op);
      if (!inst)
        return;

      inst->setMetadata(key, md);
    }
  }

  LogicalResult
  amendOperation(Operation *op, NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const override {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
      // auto *fn = moduleTranslation.lookupFunction(funcOp.getName());
      // errs() << "[translation] fn: " << *fn << "\n";
      return success();
    }

    auto iciAttr = op->getAttrOfType<BoolAttr>("enzyme.ici");
    auto icvAttr = op->getAttrOfType<BoolAttr>("enzyme.icv");

    // Op was already processed
    if (!(iciAttr && icvAttr))
      return success();

    // Convert the attributes to the appropriate metadata
    if (iciAttr.getValue() && icvAttr.getValue())
      annotateActivity(op, "enzyme_inactive", moduleTranslation);
    else if (!iciAttr.getValue() && !icvAttr.getValue())
      annotateActivity(op, "enzyme_active", moduleTranslation);
    else {
      StringRef instActivity =
          iciAttr.getValue() ? "enzyme_inactive_inst" : "enzyme_active_inst";
      StringRef valActivity =
          icvAttr.getValue() ? "enzyme_inactive_val" : "enzyme_active_val";
      annotateActivity(op, instActivity, moduleTranslation);
      annotateActivity(op, valActivity, moduleTranslation);
    }

    op->removeAttr("enzyme.ici");
    op->removeAttr("enzyme.icv");
    return success();
  }
};

int main(int argc, char **argv) {
  mlir::registerAllTranslations();

  mlir::TranslateFromMLIRRegistration withdescription(
      "activity-to-llvm", "different from option",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](mlir::DialectRegistry &registry) {
        registry
            .insert<DLTIDialect, func::FuncDialect, enzyme::EnzymeDialect>();
        registerAllToLLVMIRTranslations(registry);
        registry.addExtension(
            +[](MLIRContext *ctx, enzyme::EnzymeDialect *dialect) {
              dialect->addInterfaces<ActivityToMetadataTranslation>();
            });
      });

  return failed(mlir::mlirTranslateMain(
      argc, argv, "Enzyme MLIR Translation Testing Tool"));
}
