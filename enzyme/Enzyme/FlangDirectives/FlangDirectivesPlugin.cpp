//===- FlangDirectivesPlugin.cpp - flang -load plugin for LLVM Enzyme -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FlangEnzymeDirectives: loaded into flang with
//
//   flang -Xflang -load -Xflang FlangEnzymeDirectives-<v>.so ...
//
// its static initializer
//   - defines the !DIR$ ENZYME directives (flang/Support/PluginDirectives.h),
//     also spelled !$enzyme, which flang then parses, resolves and lowers to
//     `fir.directives`, and
//   - adds two passes to flang's pipeline (fir::registerPassPipelineConfigCallback):
//     enzyme-fortran-directives, which turns the directives into the
//     registrations LLVM Enzyme reads, and enzyme-fir-type-annotations, which
//     carries the Fortran types LLVM IR erases to LLVM Enzyme's type analysis.
//
// LLVM Enzyme itself runs later: with -fpass-plugin=FlangEnzyme-<v>.so in
// flang, or in the (LTO) link with LLDEnzyme-<v>.so. Only the code here is in
// the plugin; MLIR, FIR and flang symbols resolve from the host at load time.
//
//===----------------------------------------------------------------------===//

#include "FlangDirectives.h"

#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Support/PluginDirectives.h"
#include "flang/Tools/CrossToolHelpers.h"

#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"

namespace {

static llvm::cl::opt<bool> typeAnnotations(
    "enzyme-fir-type-annotations", llvm::cl::init(true),
    llvm::cl::desc("Carry the Fortran types that LLVM IR erases to LLVM "
                   "Enzyme's type analysis (all kinds; each kind has its own "
                   "-enzyme-fir-{arg,common,runtime,literal}-types)"));

// The !DIR$ ENZYME directives (see FortranDirectives.cpp).
static void registerEnzymeDirectives() {
  using namespace Fortran::common;
  auto procArg = [](const char *keyword) {
    return PluginDirectiveArg{keyword, PluginDirectiveArgKind::Procedure};
  };
  registerPluginDirective({"enzyme", "inactive", PluginDirectiveSubject::Any});
  registerPluginDirective({"enzyme", "no_escaping_allocation",
                           PluginDirectiveSubject::Procedure});
  registerPluginDirective(
      {"enzyme",
       "custom_rule",
       PluginDirectiveSubject::Procedure,
       {procArg("forward"), procArg("augmented"), procArg("reverse")}});
  registerPluginDirective(
      {"enzyme",
       "shadow",
       PluginDirectiveSubject::Variable,
       {PluginDirectiveArg{"shadow", PluginDirectiveArgKind::Variable,
                           /*required=*/true}}});
  // In front of a DO or DO WHILE loop: the loop iterates its variables to a
  // fixed point (__enzyme_set_fixed_point).
  registerPluginDirective(
      {"enzyme",
       "fixed_point",
       PluginDirectiveSubject::Loop,
       {PluginDirectiveArg{"reduction", PluginDirectiveArgKind::Real},
        PluginDirectiveArg{"max_iters", PluginDirectiveArgKind::Integer},
        procArg("control")},
       /*minPositional=*/1});
  // `!$enzyme ...` is the same as `!DIR$ ENZYME ...`, and a comment to
  // compilers without this plugin, as Tapenade's `!$AD` is.
  registerPluginDirectiveSentinel("enzyme");
}

struct EnzymeFlangDirectivesRegistration {
  EnzymeFlangDirectivesRegistration() {
    registerEnzymeDirectives();
    fir::registerPassPipelineConfigCallback(
        [](MLIRToLLVMPassPipelineConfig &config) {
          // The directives are on the functions and globals of the module
          // from lowering on; turn them into registrations early, before
          // anything could remove a declaration only they refer to.
          config.registerHLFIROptEarlyEPCallbacks(
              [](mlir::PassManager &pm, llvm::OptimizationLevel) {
                pm.addPass(mlir::enzyme::createFortranDirectivesPass());
              });
          // FIR still has the Fortran types (and the COMMON storage of each
          // declare) at the end of the FIR pipeline.
          config.registerFIROptLastEPCallbacks(
              [](mlir::PassManager &pm, llvm::OptimizationLevel) {
                if (typeAnnotations)
                  pm.addPass(mlir::enzyme::createFIRTypeAnnotationsPass());
              });
        });
  }
};
// Runs when `flang -fc1 -load` dlopens this object.
static EnzymeFlangDirectivesRegistration enzymeFlangDirectivesRegistration;

} // namespace
