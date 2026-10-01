//===- FIRTypeAnnotations.cpp - Fortran types for LLVM Enzyme -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Carries the type information that FIR has and LLVM IR lacks to Enzyme's
// LLVM-level type analysis, which already reads
//   - `!enzyme_type` metadata on globals (the layout of their memory) and on
//     instructions (the type of the value), and
//   - `"enzyme_type"` string attributes on function and call parameters,
// in the TypeTree syntax `{[-1]:Pointer, [-1,0]:Float@double, ...}`.
//
// The enzyme-fir-type-annotations pass runs at flang's FIROptLast extension
// point, while FIR still knows the Fortran types (and the declares still name
// the COMMON storage of each variable), and attaches `enzyme.type` (a TypeTree
// string) to the operations that survive code generation. The LLVM IR
// translation interface of the (op-less) enzyme dialect, which the pass
// loads, turns them into the metadata and attributes above when flang
// translates to LLVM IR.
// Nothing here needs a change to flang.
//
// Annotated so far:
//   - COMMON blocks: the scalar type at each member offset, from all the
//     declares of the unit, if they lay out all of the block and agree.
//     This types, e.g., a memset that memcpyopt fused from the stores to
//     adjacent members.
//   - Character literals (_QQcl*): character data.
//
//===----------------------------------------------------------------------===//

#include "FlangDirectives.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

#include <map>
#include <optional>
#include <set>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// TypeTree strings
//===----------------------------------------------------------------------===//

// A TypeTree as Enzyme writes it: a map from an offset path to a base type
// name ("Pointer", "Integer", "Float@double", ...).
using TypePaths = std::map<std::vector<int>, std::string>;

static std::string printTypeTree(const TypePaths &tree) {
  std::string out = "{";
  bool first = true;
  for (const auto &[path, type] : tree) {
    if (!first)
      out += ", ";
    first = false;
    out += "[";
    for (size_t i = 0; i < path.size(); ++i)
      out += (i ? "," : "") + std::to_string(path[i]);
    out += "]:" + type;
  }
  return out + "}";
}

// Parse `{[-1]:Pointer, [-1,0]:Float@double}`.
static std::optional<TypePaths> parseTypeTree(StringRef str) {
  TypePaths tree;
  str = str.trim();
  if (!str.consume_front("{") || !str.consume_back("}"))
    return std::nullopt;
  while (!(str = str.trim()).empty()) {
    if (!str.consume_front("["))
      return std::nullopt;
    auto [pathStr, rest] = str.split(']');
    std::vector<int> path;
    for (StringRef p = pathStr; !p.empty();) {
      auto [off, tail] = p.split(',');
      int v;
      if (off.trim().getAsInteger(10, v))
        return std::nullopt;
      path.push_back(v);
      p = tail;
    }
    if (!rest.consume_front(":"))
      return std::nullopt;
    auto [type, tail] = rest.split(',');
    // "Float@double" contains no comma; the next entry starts with '['.
    tree[path] = type.trim().str();
    str = tail;
  }
  return tree;
}

// The metadata encoding of TypeTree::toMD: !{!"<type at []>", i32 off,
// <subtree>, ...}, grouped by the first offset of each path.
static llvm::MDNode *typeTreeToMD(const TypePaths &tree,
                                  llvm::LLVMContext &ctx) {
  std::string base = "Unknown";
  std::map<int, TypePaths> children;
  for (const auto &[path, type] : tree) {
    if (path.empty()) {
      base = type;
      continue;
    }
    children[path.front()][std::vector<int>(path.begin() + 1, path.end())] =
        type;
  }
  SmallVector<llvm::Metadata *> ops{llvm::MDString::get(ctx, base)};
  for (const auto &[off, sub] : children) {
    ops.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
        llvm::IntegerType::get(ctx, 32), off, /*IsSigned=*/true)));
    ops.push_back(typeTreeToMD(sub, ctx));
  }
  return llvm::MDNode::get(ctx, ops);
}

//===----------------------------------------------------------------------===//
// Translation to LLVM IR
//===----------------------------------------------------------------------===//

constexpr StringLiteral kTypeAttr = "enzyme.type";
constexpr StringLiteral kArgTypesAttr = "enzyme.arg_types";

struct EnzymeLLVMIRTranslation : public LLVMTranslationDialectInterface {
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  // enzyme.type on a global: the layout of its memory. On another op: the
  // type of the value of each instruction it became.
  // enzyme.arg_types on a call: the "enzyme_type" of each argument.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    if (attribute.getName() == kArgTypesAttr) {
      auto types = dyn_cast<ArrayAttr>(attribute.getValue());
      if (!types)
        return op->emitError() << kArgTypesAttr << " must be an array";
      for (llvm::Instruction *inst : instructions)
        if (auto *call = dyn_cast<llvm::CallBase>(inst))
          for (auto [i, t] : llvm::enumerate(types))
            if (auto s = dyn_cast<StringAttr>(t);
                s && !s.empty() && i < call->arg_size())
              call->addParamAttr(i, llvm::Attribute::get(call->getContext(),
                                                         "enzyme_type",
                                                         s.getValue()));
      return success();
    }
    if (attribute.getName() != kTypeAttr)
      return success();
    auto str = dyn_cast<StringAttr>(attribute.getValue());
    std::optional<TypePaths> tree =
        str ? parseTypeTree(str.getValue()) : std::nullopt;
    if (!tree)
      return op->emitError()
             << "malformed " << kTypeAttr << ": " << attribute.getValue();
    if (isa<LLVM::GlobalOp>(op)) {
      if (auto *gv = dyn_cast_or_null<llvm::GlobalVariable>(
              moduleTranslation.lookupGlobal(op)))
        gv->setMetadata("enzyme_type", typeTreeToMD(*tree, gv->getContext()));
      return success();
    }
    for (llvm::Instruction *inst : instructions)
      inst->setMetadata("enzyme_type", typeTreeToMD(*tree, inst->getContext()));
    return success();
  }

  // enzyme.type on a function argument: its "enzyme_type" attribute.
  LogicalResult
  convertParameterAttr(LLVM::LLVMFuncOp function, int argIdx,
                       NamedAttribute attribute,
                       LLVM::ModuleTranslation &moduleTranslation) const final {
    if (attribute.getName() != kTypeAttr)
      return success();
    auto str = dyn_cast<StringAttr>(attribute.getValue());
    llvm::Function *fn = moduleTranslation.lookupFunction(function.getName());
    if (!str || !fn)
      return success();
    fn->addParamAttr(
        argIdx,
        llvm::Attribute::get(fn->getContext(), "enzyme_type", str.getValue()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FIR types
//===----------------------------------------------------------------------===//

// The TypeTree name and size of a scalar Fortran type, if it has one.
static std::optional<std::pair<std::string, int64_t>>
scalarType(Type ty, const DataLayout &dl) {
  if (auto ft = dyn_cast<FloatType>(ty)) {
    std::string name = ft.isF32()   ? "float"
                       : ft.isF64() ? "double"
                       : ft.isF16() ? "half"
                                    : "";
    if (name.empty())
      return std::nullopt;
    return std::make_pair("Float@" + name, (int64_t)ft.getWidth() / 8);
  }
  if (auto it = dyn_cast<IntegerType>(ty))
    return std::make_pair(std::string("Integer"), (int64_t)it.getWidth() / 8);
  if (auto lt = dyn_cast<fir::LogicalType>(ty))
    return std::make_pair(std::string("Integer"), (int64_t)lt.getFKind());
  return std::nullopt;
}

// The size in bytes of the types addLayout handles.
static std::optional<int64_t> sizeOf(Type ty, const DataLayout &dl) {
  if (auto ct = dyn_cast<mlir::ComplexType>(ty)) {
    auto part = scalarType(ct.getElementType(), dl);
    return part ? std::optional<int64_t>(2 * part->second) : std::nullopt;
  }
  if (auto ch = dyn_cast<fir::CharacterType>(ty))
    return ch.hasConstantLen()
               ? std::optional<int64_t>(ch.getLen() * ch.getFKind())
               : std::nullopt;
  if (auto seq = dyn_cast<fir::SequenceType>(ty)) {
    if (seq.hasDynamicExtents())
      return std::nullopt;
    auto ele = sizeOf(seq.getEleTy(), dl);
    if (!ele)
      return std::nullopt;
    int64_t n = *ele;
    for (int64_t e : seq.getShape())
      n *= e;
    return n;
  }
  auto s = scalarType(ty, dl);
  return s ? std::optional<int64_t>(s->second) : std::nullopt;
}

// Add the scalar types of a value of type `ty` at `offset` to `layout`.
// Arrays are expanded element by element up to `budget` entries.
static bool addLayout(Type ty, int64_t offset, const DataLayout &dl,
                      std::map<int64_t, std::string> &layout, int64_t &budget) {
  if (auto ct = dyn_cast<mlir::ComplexType>(ty)) {
    auto part = scalarType(ct.getElementType(), dl);
    if (!part)
      return false;
    layout[offset] = part->first;
    layout[offset + part->second] = part->first;
    budget -= 2;
    return budget >= 0;
  }
  if (auto ch = dyn_cast<fir::CharacterType>(ty)) {
    // Character data: bytes (or kind-sized units) of Integer.
    if (!ch.hasConstantLen())
      return false;
    int64_t n = ch.getLen() * ch.getFKind();
    for (int64_t i = 0; i < n; i += ch.getFKind())
      layout[offset + i] = "Integer";
    budget -= n;
    return budget >= 0;
  }
  if (auto seq = dyn_cast<fir::SequenceType>(ty)) {
    if (seq.hasDynamicExtents())
      return false;
    Type ele = seq.getEleTy();
    std::optional<int64_t> eleSizeOpt = sizeOf(ele, dl);
    if (!eleSizeOpt || *eleSizeOpt <= 0)
      return false;
    int64_t eleSize = *eleSizeOpt;
    int64_t count = 1;
    for (int64_t e : seq.getShape())
      count *= e;
    for (int64_t i = 0; i < count; ++i)
      if (!addLayout(ele, offset + i * eleSize, dl, layout, budget))
        return false;
    return true;
  }
  auto s = scalarType(ty, dl);
  if (!s)
    return false;
  layout[offset] = s->first;
  return --budget >= 0;
}

// The fir.global that `v` addresses, through converts and coordinates.
static fir::GlobalOp getStorageGlobal(Value v, SymbolTable &symbols) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto addr = dyn_cast<fir::AddrOfOp>(def))
      return symbols.lookup<fir::GlobalOp>(addr.getSymbol().getRootReference());
    if (auto cvt = dyn_cast<fir::ConvertOp>(def)) {
      v = cvt.getValue();
      continue;
    }
    return {};
  }
  return {};
}

// The scalar TypeTree name of the data of a FIR entity type: the element
// type of an array, box, or address; Integer for character and logical data.
static std::optional<std::string> dataType(Type ty, const DataLayout &dl) {
  ty = fir::unwrapPassByRefType(fir::unwrapRefType(ty));
  if (auto box = dyn_cast<fir::BaseBoxType>(ty))
    ty = box.getEleTy();
  ty = fir::unwrapSequenceType(fir::unwrapRefType(ty));
  if (isa<fir::CharacterType>(ty))
    return std::string("Integer");
  if (auto ct = dyn_cast<mlir::ComplexType>(ty))
    ty = ct.getElementType();
  if (auto s = scalarType(ty, dl))
    return s->first;
  return std::nullopt;
}

// The TypeTree of a pointer argument whose FIR type (before the conversions
// for the call) is `ty`, or "" if there is nothing to add to the LLVM type.
// A descriptor ({ptr base, i64 elem_len, i32 version, i8 rank, i8 type,
// i8 attribute, i8 extra, [rank x [3 x i64]] dims}) is typed field by field,
// with the type of the data its base address points to.
static std::string pointerArgType(Type ty, const DataLayout &dl) {
  Type pointee = fir::unwrapRefType(ty);
  TypePaths tree{{{-1}, "Pointer"}};
  if (auto box = dyn_cast<fir::BaseBoxType>(pointee)) {
    tree[{-1, 0}] = "Pointer";
    if (auto data = dataType(box, dl))
      tree[{-1, 0, -1}] = *data;
    for (int off : {8, 16, 20, 21, 22, 23})
      tree[{-1, off}] = "Integer";
    if (auto seq =
            dyn_cast<fir::SequenceType>(fir::unwrapRefType(box.getEleTy())))
      if (!seq.hasUnknownShape())
        for (unsigned i = 0; i < 3 * seq.getDimension(); ++i)
          tree[{-1, (int)(24 + 8 * i)}] = "Integer";
    return printTypeTree(tree);
  }
  if (!fir::isa_ref_type(ty))
    return "";
  if (auto data = dataType(pointee, dl)) {
    tree[{-1, -1}] = *data;
    return printTypeTree(tree);
  }
  return "";
}

// The FIR type a call operand had before flang converted it for the call
// (e.g. !fir.box<!fir.array<?xf32>> before !fir.box<none>).
static Type originalType(Value v) {
  while (auto cvt = v.getDefiningOp<fir::ConvertOp>())
    v = cvt.getValue();
  return v.getType();
}

// How many layout entries one COMMON block may get; a block of large arrays
// stays unannotated rather than blowing up the metadata.
static constexpr int64_t kLayoutBudget = 1 << 14;

struct FIRTypeAnnotationsPass
    : public PassWrapper<FIRTypeAnnotationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRTypeAnnotationsPass)

  StringRef getArgument() const final { return "enzyme-fir-type-annotations"; }
  StringRef getDescription() const final {
    return "Annotate FIR with the types LLVM Enzyme's type analysis reads "
           "(enzyme.type, translated to !enzyme_type)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // Loading the enzyme dialect with its translation interface makes the
    // later translation to LLVM IR turn enzyme.* attributes into Enzyme's
    // metadata and attributes.
    registry.insert<enzyme::EnzymeAttrDialect, fir::FIROpsDialect>();
    registry.addExtension(+[](MLIRContext *, enzyme::EnzymeAttrDialect *dialect) {
      dialect->addInterfaces<EnzymeLLVMIRTranslation>();
    });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    DataLayout dl(module);
    MLIRContext *ctx = &getContext();

    // COMMON (and other storage-associated) globals: the member layout from
    // every declare naming the global as its storage.
    // The layout is all or nothing: Enzyme's type analysis takes a type that
    // is known at many offsets of a block as the type of the whole block
    // (e.g. Float@double for a block whose first member is a large REAL*8
    // array and whose second, beyond the budget, a REAL*4 one). So a block
    // is annotated only if the declares of the unit lay out all of its bytes,
    // and agree.
    struct Layout {
      std::map<int64_t, std::string> types;
      // [begin, end) of the bytes the declares lay out
      std::map<int64_t, int64_t> covered;
      bool unknown = false;
    };
    std::map<Operation *, Layout> layouts;
    module.walk([&](fir::FortranVariableStorageOpInterface decl) {
      Value storage = decl.getStorage();
      if (!storage)
        return;
      fir::GlobalOp global = getStorageGlobal(storage, symbols);
      if (!global)
        return;
      Layout &layout = layouts[global];
      if (layout.unknown)
        return;
      std::map<int64_t, std::string> member;
      int64_t budget = kLayoutBudget;
      Type ty = fir::unwrapRefType(decl->getResult(0).getType());
      int64_t offset = decl.getStorageOffset();
      std::optional<int64_t> size = sizeOf(ty, dl);
      // A member that cannot be laid out (dynamic size, derived type, too
      // large) leaves the block unknown.
      if (!size || !addLayout(ty, offset, dl, member, budget)) {
        layout.unknown = true;
        return;
      }
      for (auto &[off, t] : member) {
        auto [it, inserted] = layout.types.insert({off, t});
        if (!inserted && it->second != t) {
          layout.unknown = true;
          return;
        }
      }
      int64_t &end = layout.covered[offset];
      end = std::max(end, offset + *size);
    });
    for (auto &[op, layout] : layouts) {
      if (layout.unknown)
        continue;
      std::optional<int64_t> globalSize =
          sizeOf(cast<fir::GlobalOp>(op).getType(), dl);
      int64_t reached = 0;
      for (auto [begin, end] : layout.covered) {
        if (begin > reached)
          break;
        reached = std::max(reached, end);
      }
      if (!globalSize || reached < *globalSize)
        continue;
      TypePaths tree{{{-1}, "Pointer"}};
      for (auto &[off, t] : layout.types)
        tree[{-1, (int)off}] = t;
      if (tree.size() > 1)
        op->setAttr(kTypeAttr, StringAttr::get(ctx, printTypeTree(tree)));
    }

    // Calls to the flang runtime: the types of their pointer arguments (data,
    // descriptors, character data), which the runtime's C signatures erase.
    module.walk([&](fir::CallOp call) {
      auto callee = call.getCallee();
      if (!callee ||
          !callee->getLeafReference().getValue().starts_with("_Fortran"))
        return;
      SmallVector<Attribute> types;
      bool any = false;
      for (Value arg : call.getArgs()) {
        // Only what the conversions for the call erased: an operand of its
        // own type (e.g. an I/O cookie, !fir.ref<i8>) is opaque.
        Type original = originalType(arg);
        std::string t =
            original != arg.getType() ||
                    isa<fir::BaseBoxType>(fir::unwrapRefType(original))
                ? pointerArgType(original, dl)
                : "";
        any |= !t.empty();
        types.push_back(StringAttr::get(ctx, t));
      }
      if (any)
        call->setAttr(kArgTypesAttr, ArrayAttr::get(ctx, types));
    });

    // Character literals: character data throughout.
    for (fir::GlobalOp global : module.getOps<fir::GlobalOp>())
      if (isa<fir::CharacterType>(global.getType()) &&
          !global->hasAttr(kTypeAttr))
        global->setAttr(
            kTypeAttr, StringAttr::get(ctx, "{[-1]:Pointer, [-1,-1]:Integer}"));
  }
};

} // namespace

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::enzyme::EnzymeAttrDialect)

std::unique_ptr<Pass> mlir::enzyme::createFIRTypeAnnotationsPass() {
  return std::make_unique<FIRTypeAnnotationsPass>();
}

void mlir::enzyme::registerFIRTypeAnnotationsPass() {
  PassRegistration<FIRTypeAnnotationsPass>();
}
