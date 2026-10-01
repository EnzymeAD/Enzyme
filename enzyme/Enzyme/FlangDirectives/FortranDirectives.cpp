//===- FortranDirectives.cpp - !DIR$ ENZYME directives --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fortran source directives for Enzyme, defined through flang's plugin
// directives (flang/Support/PluginDirectives.h). flang resolves their names to
// symbols, keeps them in module files, and lowers them to a `fir.directives`
// attribute on their subject. The enzyme-fortran-directives pass turns them
// into what Enzyme reads, which works for both the MLIR and the LLVM route:
//
//   !dir$ enzyme inactive [(proc)]       llvm.passthrough "enzyme_inactive",
//                                        __enzyme_inactivefn and
//                                        __enzyme_nofree registrations
//   !dir$ enzyme inactive(var | /blk/)   __enzyme_inactive_global registration
//   !dir$ enzyme no_escaping_allocation [(proc)]
//                                        __enzyme_no_escaping_allocation
//                                        registration (no allocation of the
//                                        procedure outlives it)
//   !dir$ enzyme shadow(var, shadow=s)   __enzyme_shadow_global pair {&var, &s}
//   !dir$ enzyme shadow(/blk/, shadow=/blk_d/)  the same, block to block
//   !dir$ enzyme custom_rule [(proc)] (augmented=a, reverse=r)
//                                        __enzyme_register_gradient {p, a, r}
//   !dir$ enzyme custom_rule [(proc)] (forward=f)
//                                        __enzyme_register_derivative {p, f}
//
//===----------------------------------------------------------------------===//

#include "FlangDirectives.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

// Append `entry` to the llvm.passthrough attribute of `fn`, which becomes the
// string function attributes of the llvm.func.
static void addPassthrough(Operation *fn, StringRef entry) {
  auto *ctx = fn->getContext();
  SmallVector<Attribute> entries;
  if (auto existing = fn->getAttrOfType<ArrayAttr>("llvm.passthrough"))
    entries.append(existing.begin(), existing.end());
  for (Attribute a : entries)
    if (auto s = dyn_cast<StringAttr>(a); s && s.getValue() == entry)
      return;
  entries.push_back(StringAttr::get(ctx, entry));
  fn->setAttr("llvm.passthrough", ArrayAttr::get(ctx, entries));
}

// Create a global named `name` whose initializer is a tuple of the addresses
// of `symbols` (functions as procedure pointers, globals as raw pointers), as
// Enzyme's registration markers expect.
static LogicalResult createRegistration(ModuleOp module, Location loc,
                                        StringRef name,
                                        ArrayRef<FlatSymbolRefAttr> symbols) {
  if (module.lookupSymbol(name))
    return success();
  MLIRContext *ctx = module.getContext();
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(module.getBody());
  SmallVector<Type> types;
  for (FlatSymbolRefAttr sym : symbols) {
    Operation *op = module.lookupSymbol(sym);
    if (isa_and_nonnull<func::FuncOp>(op))
      types.push_back(
          fir::BoxProcType::get(ctx, FunctionType::get(ctx, {}, {})));
    else if (isa_and_nonnull<fir::GlobalOp>(op))
      types.push_back(fir::LLVMPointerType::get(ctx, IntegerType::get(ctx, 8)));
    else
      return emitError(loc) << "no function or global " << sym;
  }
  Type tupleTy = TupleType::get(ctx, types);
  auto global = fir::GlobalOp::create(
      b, loc, name, /*isConstant=*/false,
      /*isTarget=*/false, tupleTy,
      // Every unit that sees the directive (through a module file too) emits
      // it. Weak, not internal: nothing refers to it, and with LTO it has to
      // survive the optimization before the link, where Enzyme runs.
      fir::LinkageAttr::get(ctx, fir::LinkageEnum::Weak));
  Region &region = global.getRegion();
  Block *init = region.empty() ? b.createBlock(&region) : &region.front();
  b.setInsertionPointToStart(init);
  Value tuple = fir::UndefOp::create(b, loc, tupleTy);
  for (auto [i, sym] : llvm::enumerate(symbols)) {
    Operation *op = module.lookupSymbol(sym);
    Value elt;
    if (auto fn = dyn_cast<func::FuncOp>(op)) {
      Value addr = fir::AddrOfOp::create(b, loc, fn.getFunctionType(), sym);
      elt = fir::EmboxProcOp::create(b, loc, types[i], addr);
    } else {
      auto g = cast<fir::GlobalOp>(op);
      Value addr = fir::AddrOfOp::create(
          b, loc, fir::ReferenceType::get(g.getType()), sym);
      elt = fir::ConvertOp::create(b, loc, types[i], addr);
    }
    tuple = fir::InsertValueOp::create(
        b, loc, tupleTy, tuple, elt,
        b.getArrayAttr({b.getIntegerAttr(b.getIndexType(), i)}));
  }
  fir::HasValueOp::create(b, loc, tuple);
  return success();
}

// The name of a registration: `__enzyme_<kind>.<subject>`. Unlike a name
// starting with the subject's, flang's external name conversion leaves it
// alone (it would make `_QMmEv.__enzyme_x` `_QMmEvX__enzyme_x`), so that a
// link can keep all of them with --undefined-glob='__enzyme_*'.
static std::string registrationName(StringRef kind, StringRef subject) {
  return (kind + "." + subject).str();
}

static FlatSymbolRefAttr getSymbolArg(DictionaryAttr args, StringRef key) {
  return args ? dyn_cast_or_null<FlatSymbolRefAttr>(args.get(key)) : nullptr;
}

static LogicalResult lowerDirective(ModuleOp module, Operation *subject,
                                    DictionaryAttr directive) {
  auto keyword = directive.getAs<StringAttr>("keyword");
  auto args = directive.getAs<DictionaryAttr>("args");
  auto subjectSym = FlatSymbolRefAttr::get(SymbolTable::getSymbolName(subject));
  StringRef subjectName = SymbolTable::getSymbolName(subject).getValue();
  Location loc = subject->getLoc();

  if (keyword == "no_escaping_allocation") {
    if (!isa<func::FuncOp>(subject))
      return emitError(loc)
             << "enzyme no_escaping_allocation applies to a procedure";
    return createRegistration(
        module, loc,
        registrationName("__enzyme_no_escaping_allocation", subjectName),
        {subjectSym});
  }
  if (keyword == "inactive") {
    if (isa<func::FuncOp>(subject)) {
      // Read by Enzyme-MLIR; LLVM Enzyme reads the registration, which also
      // keeps the procedure from being inlined before it differentiates.
      addPassthrough(subject, "enzyme_inactive");
      // Inlined (with LTO, before Enzyme runs), its body would be
      // differentiated like the caller's.
      addPassthrough(subject, "noinline");
      // An inactive procedure frees nothing Enzyme has to track either.
      if (failed(createRegistration(
              module, loc, registrationName("__enzyme_nofree", subjectName),
              {subjectSym})))
        return failure();
      return createRegistration(
          module, loc, registrationName("__enzyme_inactivefn", subjectName),
          {subjectSym});
    }
    return createRegistration(
        module, loc, registrationName("__enzyme_inactive_global", subjectName),
        {subjectSym});
  }
  if (keyword == "shadow") {
    FlatSymbolRefAttr shadow = getSymbolArg(args, "shadow");
    if (!shadow || !isa<fir::GlobalOp>(subject))
      return emitError(loc) << "enzyme shadow needs a global and its shadow";
    return createRegistration(
        module, loc, registrationName("__enzyme_shadow_global", subjectName),
        {subjectSym, shadow});
  }
  if (keyword == "custom_rule") {
    if (!isa<func::FuncOp>(subject))
      return emitError(loc) << "enzyme custom_rule applies to a procedure";
    // Enzyme sees the calls only if they survive until it runs, which with
    // LTO is after the optimization of each unit.
    addPassthrough(subject, "noinline");
    FlatSymbolRefAttr forward = getSymbolArg(args, "forward");
    FlatSymbolRefAttr augmented = getSymbolArg(args, "augmented");
    FlatSymbolRefAttr reverse = getSymbolArg(args, "reverse");
    if ((augmented == nullptr) != (reverse == nullptr))
      return emitError(loc) << "enzyme custom_rule needs both augmented= and "
                               "reverse= for reverse mode";
    if (!forward && !reverse)
      return emitError(loc) << "enzyme custom_rule needs forward=, or "
                               "augmented= and reverse=";
    if (forward &&
        failed(createRegistration(
            module, loc,
            registrationName("__enzyme_register_derivative", subjectName),
            {subjectSym, forward})))
      return failure();
    if (reverse &&
        failed(createRegistration(
            module, loc,
            registrationName("__enzyme_register_gradient", subjectName),
            {subjectSym, augmented, reverse})))
      return failure();
    return success();
  }
  return emitError(loc) << "unknown enzyme directive " << keyword;
}

struct FortranDirectivesPass
    : public PassWrapper<FortranDirectivesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FortranDirectivesPass)

  StringRef getArgument() const final { return "enzyme-fortran-directives"; }
  StringRef getDescription() const final {
    return "Turn !DIR$ ENZYME directives (fir.directives) into the markers "
           "Enzyme reads";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<fir::FIROpsDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<std::pair<Operation *, DictionaryAttr>> work;
    for (Operation &op : module.getBody()->getOperations())
      if (auto dirs = op.getAttrOfType<ArrayAttr>("fir.directives"))
        for (Attribute d : dirs)
          if (auto dict = dyn_cast<DictionaryAttr>(d))
            if (dict.getAs<StringAttr>("prefix") == "enzyme")
              work.push_back({&op, dict});
    for (auto [op, dict] : work)
      if (failed(lowerDirective(module, op, dict)))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::enzyme::createFortranDirectivesPass() {
  return std::make_unique<FortranDirectivesPass>();
}

void mlir::enzyme::registerFortranDirectivesPass() {
  PassRegistration<FortranDirectivesPass>();
}
