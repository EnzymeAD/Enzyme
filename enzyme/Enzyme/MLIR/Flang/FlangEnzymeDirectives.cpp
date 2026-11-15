//===- FlangEnzymeDirectives.cpp - !DIR$ ENZYME in flang ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The !DIR$ ENZYME directives, also spelled !$enzyme, defined with flang's
// plugin directives (flang/Support/PluginDirectives.h): this file parses
// them; flang resolves their names, checks their kinds, keeps them in module
// files and lowers each onto the operation of its subject, where this file
// attaches it as an `enzyme.<keyword>` attribute: a unit attribute, or the
// dictionary of its arguments. The enzyme-fortran-directives pass
// (FortranDirectives.cpp) turns those into the registrations Enzyme reads.
//
// Built into FlangEnzymeMLIR only if flang has plugin directives. Flang loads
// FlangEnzymeMLIR (with -load, or through FlangEnzyme's loader) before it
// parses, so the directives are claimed in time.
//
//===----------------------------------------------------------------------===//

#include "Flang/FortranDirectives.h"

#include "flang/Lower/PluginDirectives.h"
#include "flang/Optimizer/Passes/Pipelines.h"
#include "flang/Support/PluginDirectives.h"
#include "flang/Tools/CrossToolHelpers.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"

namespace {

// The !DIR$ ENZYME directives (see FortranDirectives.cpp), in flang's
// call-style syntax. Flang resolves the names and checks their kinds.
static const std::vector<Fortran::common::PluginDirectiveCallStyle> &
enzymeDirectives() {
  using namespace Fortran::common;
  using Kind = PluginDirectiveValueKind;
  constexpr auto Entities = PluginDirectiveSubject::Entities;
  static const std::vector<PluginDirectiveCallStyle> forms{
      // inactive [(proc | var | /blk/ [, ...])]
      {"inactive", Entities, Kind::Entity, 0, 64, {}},
      // no_escaping_allocation [(proc)]
      {"no_escaping_allocation", Entities, Kind::Procedure, 0, 1, {}},
      // custom_rule [(proc)] (forward=f | augmented=a, reverse=r)
      {"custom_rule",
       Entities,
       Kind::Procedure,
       0,
       1,
       {{"forward", Kind::Procedure},
        {"augmented", Kind::Procedure},
        {"reverse", Kind::Procedure}}},
      // shadow(var | /blk/, shadow=var | /blk/)
      {"shadow",
       Entities,
       Kind::Variable,
       1,
       1,
       {{"shadow", Kind::Variable, /*required=*/true}}},
  };
  return forms;
}

static void
parseEnzymeDirective(std::string_view text,
                     Fortran::common::PluginDirectiveParse &result) {
  Fortran::common::parseCallStyleDirective(text, enzymeDirectives(), result);
  // Checks that need more than one argument, before flang resolves names.
  if (result.keyword != "custom_rule" || !result.errors().empty())
    return;
  bool forward = false, augmented = false, reverse = false;
  for (const auto &value : result.values) {
    forward |= value.keyword == "forward";
    augmented |= value.keyword == "augmented";
    reverse |= value.keyword == "reverse";
  }
  if (augmented != reverse)
    result.addError(text, "custom_rule needs both augmented= and reverse= "
                          "for reverse mode");
  else if (!forward && !reverse)
    result.addError(text,
                    "custom_rule needs forward=, or augmented= and reverse=");
}

static void
lowerEnzymeDirective(const Fortran::lower::LoweredPluginDirective &directive) {
  mlir::Attribute value = directive.args;
  if (directive.args.empty())
    value = mlir::UnitAttr::get(directive.op->getContext());
  directive.op->setAttr(mlir::enzyme::directiveAttrName(directive.keyword),
                        value);
}

} // namespace

void mlir::enzyme::registerFlangEnzymeDirectives() {
  // `!$enzyme ...` is the same as `!DIR$ ENZYME ...`, and a comment to
  // compilers without this plugin, as Tapenade's `!$AD` is.
  Fortran::common::registerPluginDirectives(
      {"enzyme", /*sentinel=*/true, parseEnzymeDirective});
  Fortran::lower::registerPluginDirectiveLowering("enzyme",
                                                  lowerEnzymeDirective);
  fir::registerPassPipelineConfigCallback(
      [](MLIRToLLVMPassPipelineConfig &config) {
        // The directives are on the functions and globals of the module from
        // lowering on; turn them into registrations early, before anything
        // could remove a declaration only they refer to.
        config.registerHLFIROptEarlyEPCallbacks(
            [](mlir::PassManager &pm, llvm::OptimizationLevel) {
              pm.addPass(mlir::enzyme::createFortranDirectivesPass());
            });
      });
}
