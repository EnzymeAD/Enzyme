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
// and, in front of a DO or DO WHILE loop, which flang lowers to a marker call
// at the start of the loop body (`fir.directive`):
//
//   !dir$ enzyme fixed_point(var...) [reduction(r)] [max_iters(n)]
//                            [control(proc)]
//                                        __enzyme_set_fixed_point(r, n, proc,
//                                        [&var, bytes]...) in its place
//
//===----------------------------------------------------------------------===//

#include "FlangDirectives.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/DataLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
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

// The address and size in bytes of the data of a fixed_point variable, as
// flang evaluated it in front of the loop: the address of a variable of a
// static size, or the descriptor of one (an allocatable or pointer as its
// target's, an assumed-shape or automatic array as itself).
static LogicalResult getStateRange(OpBuilder &b, Location loc, Value var,
                                   const DataLayout &dl,
                                   const fir::KindMapping &kindMap, Value &addr,
                                   Value &bytes) {
  Type ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  Type i64 = b.getI64Type();
  Type ty = var.getType();
  if (isa<fir::ReferenceType, fir::HeapType, fir::PointerType>(ty)) {
    Type eleTy = fir::dyn_cast_ptrEleTy(ty);
    auto size = fir::getTypeSizeAndAlignment(loc, eleTy, dl, kindMap);
    if (!size || fir::hasDynamicSize(eleTy))
      return emitError(loc)
             << "enzyme fixed_point: the size of " << ty << " is not known";
    addr = fir::ConvertOp::create(b, loc, ptrTy, var);
    bytes = arith::ConstantIntOp::create(b, loc, i64, size->first);
    return success();
  }
  if (auto boxTy = dyn_cast<fir::BaseBoxType>(ty)) {
    if (boxTy.isAssumedRank())
      return emitError(loc) << "enzyme fixed_point: an assumed-rank variable "
                               "is not supported";
    unsigned rank = fir::getBoxRank(boxTy);
    addr = fir::ConvertOp::create(
        b, loc, ptrTy,
        fir::BoxAddrOp::create(b, loc, boxTy.getBaseAddressType(), var));
    bytes = fir::BoxEleSizeOp::create(b, loc, i64, var);
    Type idx = b.getIndexType();
    for (unsigned d = 0; d < rank; ++d) {
      Value dim = arith::ConstantIndexOp::create(b, loc, d);
      auto dims = fir::BoxDimsOp::create(b, loc, idx, idx, idx, var, dim);
      Value extent = arith::IndexCastOp::create(b, loc, i64, dims.getResult(1));
      bytes = arith::MulIOp::create(b, loc, bytes, extent);
    }
    return success();
  }
  return emitError(loc) << "enzyme fixed_point: a variable of type " << ty
                        << " is not supported";
}

// `!dir$ enzyme fixed_point`: replace the marker flang put at the start of
// the loop body with
//   __enzyme_set_fixed_point(double reduction, i64 max_iters, ptr control,
//                            [ptr state, i64 bytes]...)
// whose operands Enzyme wants from in front of the loop: they are computed
// right after the last of the variables, which flang evaluated in front of
// the loop. A missing reduction or max_iters is -1 (Enzyme's default), a
// missing control null (Enzyme's test).
static LogicalResult lowerFixedPoint(ModuleOp module, fir::CallOp marker,
                                     DictionaryAttr args, DominanceInfo &dom) {
  Location loc = marker.getLoc();
  MLIRContext *ctx = module.getContext();
  if (marker.getArgs().empty())
    return emitError(loc) << "enzyme fixed_point needs a variable";
  std::optional<DataLayout> dl =
      fir::support::getOrSetMLIRDataLayout(module, /*allowDefaultLayout=*/true);
  fir::KindMapping kindMap = fir::getKindMapping(module);

  Operation *last = nullptr;
  for (Value v : marker.getArgs())
    if (Operation *def = v.getDefiningOp())
      if (!last || dom.properlyDominates(last, def))
        last = def;
  OpBuilder b(ctx);
  if (last)
    b.setInsertionPointAfter(last);
  else
    b.setInsertionPointToStart(
        &marker->getParentOfType<func::FuncOp>().getBody().front());

  Type ptrTy = LLVM::LLVMPointerType::get(ctx);
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  double reduction = -1.0;
  if (Attribute r = args ? args.get("reduction") : Attribute()) {
    if (auto f = dyn_cast<FloatAttr>(r))
      reduction = f.getValueAsDouble();
    else if (auto i = dyn_cast<IntegerAttr>(r))
      reduction = static_cast<double>(i.getInt());
    else
      return emitError(loc) << "enzyme fixed_point: reduction must be a number";
  }
  int64_t maxIters = -1;
  if (auto n = args ? args.getAs<IntegerAttr>("max_iters") : IntegerAttr())
    maxIters = n.getInt();
  SmallVector<Value> operands{
      arith::ConstantFloatOp::create(b, loc, cast<FloatType>(f64),
                                     APFloat(reduction)),
      arith::ConstantIntOp::create(b, loc, i64, maxIters)};
  if (FlatSymbolRefAttr control = getSymbolArg(args, "control")) {
    auto fn = module.lookupSymbol<func::FuncOp>(control);
    if (!fn)
      return emitError(loc) << "enzyme fixed_point: control " << control
                            << " is not a procedure";
    Value addr = fir::AddrOfOp::create(b, loc, fn.getFunctionType(), control);
    operands.push_back(fir::ConvertOp::create(b, loc, ptrTy, addr));
  } else {
    operands.push_back(LLVM::ZeroOp::create(b, loc, ptrTy));
  }
  for (Value v : marker.getArgs()) {
    Value addr, bytes;
    if (failed(getStateRange(b, loc, v, *dl, kindMap, addr, bytes)))
      return failure();
    operands.push_back(addr);
    operands.push_back(bytes);
  }

  // declare void @__enzyme_set_fixed_point(double, i64, ptr, ...)
  StringRef name = "__enzyme_set_fixed_point";
  auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                          {f64, i64, ptrTy}, /*isVarArg=*/true);
  auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!callee) {
    OpBuilder mb(module.getBodyRegion());
    mb.setInsertionPointToEnd(module.getBody());
    callee = LLVM::LLVMFuncOp::create(mb, loc, name, fnTy);
  }
  b.setInsertionPoint(marker);
  LLVM::CallOp::create(b, loc, callee, operands);
  return success();
}

// The markers flang put in loops for the loop directives, replaced and
// removed, with the declarations of their callees.
static LogicalResult lowerLoopDirectives(ModuleOp module) {
  SmallVector<fir::CallOp> markers;
  module.walk([&](fir::CallOp call) {
    if (auto dict = call->getAttrOfType<DictionaryAttr>("fir.directive"))
      if (dict.getAs<StringAttr>("prefix") == "enzyme")
        markers.push_back(call);
  });
  DominanceInfo dom(module);
  SetVector<Operation *> callees;
  for (fir::CallOp marker : markers) {
    auto dict = marker->getAttrOfType<DictionaryAttr>("fir.directive");
    auto keyword = dict.getAs<StringAttr>("keyword");
    if (keyword != "fixed_point")
      return emitError(marker.getLoc())
             << "unknown enzyme loop directive " << keyword;
    if (failed(lowerFixedPoint(module, marker,
                               dict.getAs<DictionaryAttr>("args"), dom)))
      return failure();
    if (SymbolRefAttr sym = marker.getCalleeAttr())
      if (Operation *fn = module.lookupSymbol(sym))
        callees.insert(fn);
    marker->erase();
  }
  for (Operation *fn : callees)
    if (SymbolTable::symbolKnownUseEmpty(fn, module))
      fn->erase();
  return success();
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
    registry
        .insert<fir::FIROpsDialect, arith::ArithDialect, LLVM::LLVMDialect>();
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
    if (failed(lowerLoopDirectives(module)))
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
