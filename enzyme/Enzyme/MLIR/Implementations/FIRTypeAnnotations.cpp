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
// string) to the operations that survive code generation. An LLVM IR
// translation interface that the pass attaches to the enzyme dialect turns
// them into the metadata and attributes above when flang translates to LLVM
// IR. FlangEnzymeMLIR runs it in flang (HLFIRFlangPluginRegistration.cpp);
// FIREnzyme registers it for fir-opt.
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

#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"

#include "Dialect/Dialect.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/Support/InternalNames.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"

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
constexpr StringLiteral kRetTypeAttr = "enzyme.ret_type";
// On a function: the types of its local variables, by name.
constexpr StringLiteral kLocalTypesAttr = "enzyme.local_types";

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
    if (attribute.getName() == kRetTypeAttr) {
      auto str = dyn_cast<StringAttr>(attribute.getValue());
      auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (!str || !fn)
        return op->emitError() << kRetTypeAttr << " must be a string on a "
                               << "function";
      if (llvm::Function *f = moduleTranslation.lookupFunction(fn.getName()))
        if (!f->getReturnType()->isVoidTy())
          f->addRetAttr(llvm::Attribute::get(f->getContext(), "enzyme_type",
                                             str.getValue()));
      return success();
    }
    if (attribute.getName() == kLocalTypesAttr) {
      // Converted after the body of the function: its values are mapped.
      auto types = dyn_cast<DictionaryAttr>(attribute.getValue());
      if (!types)
        return success();
      op->walk([&](LLVM::AllocaOp alloca) {
        auto name = alloca->getAttrOfType<StringAttr>("bindc_name");
        auto entry = name ? types.getAs<ArrayAttr>(name) : ArrayAttr();
        if (!entry || entry.size() != 2)
          return;
        auto t = dyn_cast<StringAttr>(entry[0]);
        auto bytes = dyn_cast<IntegerAttr>(entry[1]);
        std::optional<TypePaths> tree =
            t ? parseTypeTree(t.getValue()) : std::nullopt;
        auto *inst = dyn_cast_or_null<llvm::AllocaInst>(
            moduleTranslation.lookupValue(alloca.getResult()));
        if (!tree || !bytes || !inst)
          return;
        // Only on an alloca of the size of the variable (or of a size known
        // only at run time, if the variable's is).
        std::optional<llvm::TypeSize> size =
            inst->getAllocationSize(inst->getDataLayout());
        int64_t expected = bytes.getInt();
        if (expected >= 0 ? (!size || size->isScalable() ||
                             (int64_t)size->getFixedValue() != expected)
                          : size.has_value())
          return;
        inst->setMetadata("enzyme_type",
                          typeTreeToMD(*tree, inst->getContext()));
      });
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

// The TypeTree name and size in bytes of a scalar Fortran type, if it has
// one. The size is the distance between two array elements.
static std::optional<std::pair<std::string, int64_t>>
scalarType(Type ty, const DataLayout &dl) {
  // REAL(2), REAL(3), REAL(4), REAL(8) and REAL(16): the names of
  // ConcreteType(StringRef). REAL(10) (x87 extended precision) is left
  // alone: its 10 bytes are padded to a size that depends on the target.
  if (auto ft = dyn_cast<FloatType>(ty)) {
    std::string name = ft.isF16()    ? "half"
                       : ft.isBF16() ? "bf16"
                       : ft.isF32()  ? "float"
                       : ft.isF64()  ? "double"
                       : ft.isF128() ? "fp128"
                                     : "";
    if (name.empty())
      return std::nullopt;
    return std::make_pair("Float@" + name, (int64_t)ft.getWidth() / 8);
  }
  // INTEGER of any kind is an iN of the kind's bytes. Enzyme's Integer has
  // no width: it says that the bytes it covers are not a floating-point
  // value or a pointer, so the width is only the size.
  if (auto it = dyn_cast<IntegerType>(ty))
    return std::make_pair(std::string("Integer"), (int64_t)it.getWidth() / 8);
  // Enzyme's TypeTree has no boolean type: its concrete types are Integer,
  // Float, Pointer and Anything. A LOGICAL is stored as an integer of its
  // kind's bytes, and as such it is Integer.
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

// The one scalar type of all of a value of type `ty`, if it has one: of a
// scalar, of an array of scalars, Integer for CHARACTER data. COMPLEX has
// one too (both parts are the same).
static std::optional<std::string> uniformType(Type ty, const DataLayout &dl) {
  ty = fir::unwrapSequenceType(ty);
  if (isa<fir::CharacterType>(ty))
    return std::string("Integer");
  if (auto ct = dyn_cast<mlir::ComplexType>(ty))
    ty = ct.getElementType();
  if (auto sc = scalarType(ty, dl))
    return sc->first;
  return std::nullopt;
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

// The byte offsets of the fields of a descriptor (CFI_cdesc_t of
// ISO_Fortran_binding.h, as flang lays it out for a 64-bit target):
//   { ptr base_addr, i64 elem_len, i32 version, i8 rank, i8 type,
//     i8 attribute, i8 extra, [rank x [3 x i64]] dim }
// where each dim is { lower_bound, extent, sm (the stride in bytes) }. A
// derived type's addendum after the dims is not typed.
static constexpr int kDescBaseAddr = 0;
static constexpr int kDescElemLen = 8;
static constexpr int kDescVersion = 16;
static constexpr int kDescRank = 20;
static constexpr int kDescType = 21;
static constexpr int kDescAttribute = 22;
static constexpr int kDescExtra = 23;
static constexpr int kDescDims = 24;
static constexpr int kDescDimFieldSize = 8;
static constexpr int kDescDimFields = 3;

// The TypeTree of a pointer argument whose FIR type (before the conversions
// for the call) is `ty`, or "" if there is nothing to add to the LLVM type.
// A descriptor is typed field by field (all of them Integer but the base
// address), with the type of the data its base address points to.
static std::string pointerArgType(Type ty, const DataLayout &dl,
                                  bool descriptorData = true) {
  Type pointee = fir::unwrapRefType(ty);
  TypePaths tree{{{-1}, "Pointer"}};
  if (auto box = dyn_cast<fir::BaseBoxType>(pointee)) {
    tree[{-1, kDescBaseAddr}] = "Pointer";
    if (auto data = dataType(box, dl); data && descriptorData)
      tree[{-1, kDescBaseAddr, -1}] = *data;
    for (int off : {kDescElemLen, kDescVersion, kDescRank, kDescType,
                    kDescAttribute, kDescExtra})
      tree[{-1, off}] = "Integer";
    // The dims, if the rank is known.
    if (auto seq =
            dyn_cast<fir::SequenceType>(fir::unwrapRefType(box.getEleTy())))
      if (!seq.hasUnknownShape())
        for (unsigned i = 0; i < kDescDimFields * seq.getDimension(); ++i)
          tree[{-1, (int)(kDescDims + kDescDimFieldSize * i)}] = "Integer";
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
// Enzyme's type analysis drops offsets beyond -enzyme-max-type-offset
// (default 500), and then takes the types it keeps for the whole block (e.g.
// a REAL*8 array followed by a REAL*4 one all for REAL*8). Larger blocks
// stay unannotated.
static constexpr int64_t kMaxTypeOffset = 500;

// How many scalars the pointee of a scalar dummy argument may be laid out
// as (a COMPLEX is two). Larger pointees (derived types are not laid out at
// all) stay unannotated, which keeps the attributes short.
static constexpr int64_t kMaxTypeOffsetForArgs = 16;

// Enzyme's TypeTree has no extent: a type at offset -1 of what a pointer
// points to holds at every offset from it, also beyond the object it is
// meant for. Data that may be part of a larger object of other types (a
// component of a derived type, a member of a COMMON block or EQUIVALENCE
// group, a dummy argument, whose actual may be any of these) is therefore
// typed at its offsets one by one, if its size is known and within Enzyme's
// type offsets, and not at all otherwise.
static llvm::cl::opt<bool> annotateUnbounded(
    "enzyme-fir-arg-unbounded-types", llvm::cl::init(false),
    llvm::cl::desc("Type the data of array, CHARACTER and descriptor dummy "
                   "arguments at every offset, also when their size is not "
                   "known (unsound if the actual argument is a component or "
                   "storage associated)"));

// Type the data of `bytes` bytes (if known) of scalars `t` of `eleSize`
// bytes, under `path`, of something that may be part of a larger object.
static void addBoundedData(TypePaths &tree, std::vector<int> path,
                           const std::string &t, int64_t eleSize,
                           std::optional<int64_t> bytes, bool unbounded) {
  if (unbounded) {
    path.push_back(-1);
    tree[path] = t;
    return;
  }
  if (!bytes || *bytes > kMaxTypeOffset || eleSize <= 0)
    return;
  for (int64_t off = 0; off < *bytes; off += eleSize) {
    auto p = path;
    p.push_back((int)off);
    tree[p] = t;
  }
}

static Type originalType(Value v) {
  while (auto cvt = v.getDefiningOp<fir::ConvertOp>())
    v = cvt.getValue();
  return v.getType();
}

// Whether the data that `v` (an address or a descriptor) refers to is a whole
// object: a local or global variable, or an ALLOCATABLE's allocation, of
// which no other data follows. Otherwise (a component, a member of a COMMON
// block or EQUIVALENCE group, a dummy argument, a POINTER's target, ...) the
// data may be part of a larger object of other types: MITgcm's
// /EE_BUFFERS_GLOBAL/ holds a REAL*8 and a REAL*4 buffer, ICON's
// phyProcGroup%grpName is a CHARACTER component followed by a descriptor.
static bool wholeObject(Value v) {
  for (int depth = 0; v && depth < 64; ++depth) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return false; // a dummy argument
    if (auto decl = dyn_cast<fir::FortranVariableStorageOpInterface>(def)) {
      if (decl.getStorage())
        return false;
      auto var = dyn_cast<fir::FortranVariableOpInterface>(def);
      if (var && var.isPointer())
        return false;
      if (var && var.isAllocatable())
        return true;
      Operation *mem = def->getOperand(0).getDefiningOp();
      return isa_and_nonnull<fir::AllocaOp, fir::AddrOfOp, fir::AllocMemOp>(
          mem);
    }
    if (auto op = dyn_cast<fir::ConvertOp>(def))
      v = op.getValue(); // the same address or descriptor, retyped
    else if (auto op = dyn_cast<fir::EmboxOp>(def))
      v = op.getMemref(); // a descriptor made for this address
    else if (auto op = dyn_cast<fir::ReboxOp>(def))
      v = op.getBox(); // a descriptor made from another (e.g. a section)
    else if (auto op = dyn_cast<fir::BoxAddrOp>(def))
      v = op.getVal(); // the base address of a descriptor
    else if (auto op = dyn_cast<fir::ArrayCoorOp>(def))
      v = op.getMemref(); // an element of an array: the same object
    else if (auto op = dyn_cast<fir::CoordinateOp>(def)) {
      // Into an array: still the same object; into a record: a component.
      if (isa<fir::RecordType>(fir::unwrapSequenceType(
              fir::unwrapPassByRefType(op.getRef().getType()))))
        return false;
      v = op.getRef();
    } else if (auto op = dyn_cast<fir::LoadOp>(def))
      v = op.getMemref(); // the descriptor of an ALLOCATABLE or POINTER
    else if (isa<fir::AllocaOp, fir::AddrOfOp, fir::AllocMemOp>(def))
      return true; // a variable or an allocation of its own
    else
      return false; // anything else may be part of a larger object
  }
  return false;
}

static llvm::cl::opt<bool> annotateDescriptorData(
    "enzyme-fir-arg-descriptor-data-types", llvm::cl::init(true),
    llvm::cl::desc("With -enzyme-fir-arg-types, also annotate the type of "
                   "the data that descriptor arguments point to (a caller "
                   "may reuse one descriptor temporary for several types)"));

// The TypeTree of a value of FIR type `ty` passed to or returned from a
// procedure, in the encoding Enzyme.jl uses for its arguments, or "" if
// there is nothing certain to say. By reference, the pointee is laid out from
// offset 0 ({[-1]:Pointer, [-1,0]:Float@double} for a REAL*8 scalar), an
// array of scalars at every offset ([-1,-1]), a descriptor field by field
// (as for runtime calls). Polymorphic and assumed-type/-rank entities, and
// derived types, are left alone.
static std::string procArgType(Type ty, const DataLayout &dl) {
  // By value: a scalar.
  if (!fir::isa_ref_type(ty) && !isa<fir::BaseBoxType>(ty)) {
    if (auto ct = dyn_cast<mlir::ComplexType>(ty))
      ty = ct.getElementType();
    if (auto sc = scalarType(ty, dl))
      return printTypeTree({{{-1}, sc->first}});
    return "";
  }
  Type pointee = fir::unwrapRefType(ty);
  if (auto box = dyn_cast<fir::BaseBoxType>(pointee)) {
    Type ele = fir::unwrapPassByRefType(box.getEleTy());
    if (isa<fir::ClassType>(box) || fir::isAssumedType(ty) ||
        isa<NoneType>(fir::unwrapSequenceType(ele)))
      return "";
    if (auto seq = dyn_cast<fir::SequenceType>(ele);
        seq && seq.hasUnknownShape())
      return ""; // assumed rank
    if (!dataType(box, dl))
      return "";
    // The data of a descriptor dummy may be a component (or section of one).
    return pointerArgType(ty, dl, annotateDescriptorData && annotateUnbounded);
  }
  if (!fir::isa_ref_type(ty))
    return "";
  TypePaths tree{{{-1}, "Pointer"}};
  // An array or CHARACTER dummy: its actual may be part of a larger object.
  if (isa<fir::SequenceType, fir::CharacterType>(pointee)) {
    Type ele = fir::unwrapSequenceType(pointee);
    std::optional<std::string> t = uniformType(ele, dl);
    if (!t)
      return "";
    int64_t eleSize = 1;
    if (!isa<fir::CharacterType>(ele)) {
      if (auto ct = dyn_cast<mlir::ComplexType>(ele))
        ele = ct.getElementType();
      eleSize = scalarType(ele, dl)->second;
    }
    addBoundedData(tree, {-1}, *t, eleSize, sizeOf(pointee, dl),
                   annotateUnbounded);
    return printTypeTree(tree);
  }
  std::map<int64_t, std::string> layout;
  int64_t budget = kMaxTypeOffsetForArgs;
  if (!addLayout(pointee, 0, dl, layout, budget))
    return "";
  for (auto &[off, t] : layout)
    tree[{-1, (int)off}] = t;
  return printTypeTree(tree);
}

// How many layout entries one COMMON block may get; a block of large arrays
// stays unannotated rather than blowing up the metadata.
static constexpr int64_t kLayoutBudget = 1 << 14;

// What to annotate, each on its own (all of it with
// -enzyme-fir-type-annotations, see HLFIRFlangPluginRegistration.cpp).
static llvm::cl::opt<bool> annotateLocalNumbers(
    "enzyme-fir-local-number-types", llvm::cl::init(true),
    llvm::cl::desc("With -enzyme-fir-local-types, also annotate local REAL, "
                   "INTEGER, LOGICAL and COMPLEX variables (not only "
                   "CHARACTER ones)"));

// The TypeTree of a local variable of FIR type `ty` (its alloca: Pointer to
// the data) and its size in bytes (-1 if not constant), if it has one type.
static std::optional<std::pair<std::string, int64_t>>
localType(Type ty, const DataLayout &dl) {
  Type ele = fir::unwrapSequenceType(ty);
  std::optional<std::string> t = uniformType(ele, dl);
  if (!t)
    return std::nullopt;
  std::optional<int64_t> size = sizeOf(ty, dl);
  TypePaths tree{{{-1}, "Pointer"}};
  if (isa<fir::CharacterType>(ele) || !annotateLocalNumbers) {
    if (!isa<fir::CharacterType>(ele))
      return std::nullopt;
    tree[{-1, -1}] = "Integer";
    return std::make_pair(printTypeTree(tree), size ? *size : -1);
  }
  if (auto ct = dyn_cast<mlir::ComplexType>(ele))
    ele = ct.getElementType();
  int64_t eleSize = scalarType(ele, dl)->second;
  if (size && *size <= kMaxTypeOffset)
    addBoundedData(tree, {-1}, *t, eleSize, size, /*unbounded=*/false);
  else
    tree[{-1, -1}] = *t; // a whole object: nothing follows
  return std::make_pair(printTypeTree(tree), size ? *size : -1);
}

static llvm::cl::opt<bool> annotateProcArgs(
    "enzyme-fir-arg-types", llvm::cl::init(true),
    llvm::cl::desc("Annotate the arguments and results of procedures with "
                   "their Fortran types for LLVM Enzyme (enzyme_type)"));
static llvm::cl::opt<bool> annotateCommon(
    "enzyme-fir-common-types", llvm::cl::init(true),
    llvm::cl::desc("Annotate COMMON blocks with the types of their members "
                   "for LLVM Enzyme (!enzyme_type)"));
static llvm::cl::opt<bool> annotateRuntimeCalls(
    "enzyme-fir-runtime-types", llvm::cl::init(true),
    llvm::cl::desc("Annotate the arguments of calls to the flang runtime "
                   "with their Fortran types for LLVM Enzyme (enzyme_type)"));
static llvm::cl::opt<bool> annotateLocals(
    "enzyme-fir-local-types", llvm::cl::init(true),
    llvm::cl::desc("Annotate local CHARACTER variables as character data for "
                   "LLVM Enzyme (!enzyme_type)"));
static llvm::cl::opt<bool> annotateLiterals(
    "enzyme-fir-literal-types", llvm::cl::init(true),
    llvm::cl::desc("Annotate character literals as character data for LLVM "
                   "Enzyme (!enzyme_type)"));

struct FIRTypeAnnotationsPass
    : public PassWrapper<FIRTypeAnnotationsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRTypeAnnotationsPass)

  StringRef getArgument() const final { return "enzyme-fir-type-annotations"; }
  StringRef getDescription() const final {
    return "Annotate FIR with the types LLVM Enzyme's type analysis reads "
           "(enzyme.type, translated to !enzyme_type)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // Loading the enzyme dialect with this translation interface makes the
    // later translation to LLVM IR turn the enzyme.* attributes above into
    // Enzyme's metadata and attributes (and leave the others alone).
    registry.insert<enzyme::EnzymeDialect, fir::FIROpsDialect>();
    registry.addExtension(+[](MLIRContext *, enzyme::EnzymeDialect *dialect) {
      if (!dialect->getRegisteredInterface<LLVMTranslationDialectInterface>())
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
    // The layout is all or nothing: Enzyme's type analysis takes the types
    // it knows at some offsets of a block for the whole block. So a block is
    // annotated only if the declares of the unit lay out all of its bytes,
    // and agree, and it fits in Enzyme's type offsets -- or, whatever its
    // size, if all of its members have the same scalar type (e.g. a block
    // of CHARACTER data, Integer bytes), as that type at every offset.
    struct Layout {
      std::map<int64_t, std::string> types;
      // The one scalar type of all members, if they have one (e.g. Integer
      // for CHARACTER members): a block of it can be annotated whatever its
      // size, as that type at every offset.
      std::optional<std::string> uniform;
      bool mixed = false;
      // [begin, end) of the bytes the declares lay out
      std::map<int64_t, int64_t> covered;
      bool unknown = false;
    };
    std::map<Operation *, Layout> layouts;
    module.walk([&](fir::FortranVariableStorageOpInterface decl) {
      if (!annotateCommon)
        return;
      Value storage = decl.getStorage();
      if (!storage)
        return;
      fir::GlobalOp global = getStorageGlobal(storage, symbols);
      if (!global)
        return;
      Layout &layout = layouts[global];
      std::map<int64_t, std::string> member;
      int64_t budget = kLayoutBudget;
      Type ty = fir::unwrapRefType(decl->getResult(0).getType());
      int64_t offset = decl.getStorageOffset();
      std::optional<int64_t> size = sizeOf(ty, dl);
      if (!size) {
        layout.unknown = layout.mixed = true;
        return;
      }
      int64_t &end = layout.covered[offset];
      end = std::max(end, offset + *size);
      std::optional<std::string> uniform = uniformType(ty, dl);
      if (!uniform || (layout.uniform && *layout.uniform != *uniform))
        layout.mixed = true;
      else
        layout.uniform = uniform;
      // A member that cannot be laid out (dynamic size, derived type, too
      // large) leaves the block unknown, unless it is uniform.
      if (layout.unknown || !addLayout(ty, offset, dl, member, budget)) {
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
    });
    for (auto &[op, layout] : layouts) {
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
      if (!layout.mixed && layout.uniform) {
        op->setAttr(
            kTypeAttr,
            StringAttr::get(ctx, printTypeTree({{{-1}, "Pointer"},
                                                {{-1, -1}, *layout.uniform}})));
        continue;
      }
      if (layout.unknown || *globalSize > kMaxTypeOffset)
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
      if (!annotateRuntimeCalls || !callee ||
          !callee->getLeafReference().getValue().starts_with("_Fortran"))
        return;
      SmallVector<Attribute> types;
      bool any = false;
      for (Value arg : call.getArgs()) {
        // Only what the conversions for the call erased: an operand of its
        // own type (e.g. an I/O cookie, !fir.ref<i8>) is opaque.
        Type original = originalType(arg);
        bool erased = original != arg.getType() ||
                      isa<fir::BaseBoxType>(fir::unwrapRefType(original));
        std::string t;
        if (erased && wholeObject(arg))
          t = pointerArgType(original, dl);
        else if (erased && isa<fir::BaseBoxType>(fir::unwrapRefType(original)))
          // Possibly part of a larger object: the layout of the descriptor,
          // but not the type of its data.
          t = pointerArgType(original, dl, /*descriptorData=*/false);
        any |= !t.empty();
        types.push_back(StringAttr::get(ctx, t));
      }
      if (any)
        call->setAttr(kArgTypesAttr, ArrayAttr::get(ctx, types));
    });

    // Procedures: the types of their dummy arguments and results. Module
    // procedures also where they are only declared (from the module file),
    // so that units agree; an external procedure only where it is defined,
    // since a declaration may have been made up from the actual arguments of
    // a call through an implicit interface.
    if (annotateProcArgs)
      for (auto fn : module.getOps<func::FuncOp>()) {
        if (fn.isDeclaration() &&
            !fir::NameUniquer::deconstruct(fn.getSymName())
                 .second.modules.size())
          continue;
        FunctionType fty = fn.getFunctionType();
        for (auto [i, ty] : llvm::enumerate(fty.getInputs())) {
          // A CHARACTER dummy: flang passes the address of its data here
          // (the fir.boxchar, which is typed as character data), and its
          // length as an extra argument after all the others, which is left
          // alone. With the length typed Integer, Enzyme's type analysis
          // failed on a mask of it (`and len, 0x7fffffff`, from a loop over
          // the characters of the dummy, e.g. in a function that finds its
          // last nonblank character), which it took for a possibly
          // floating-point operation. Untyped, the analysis infers Integer
          // from how the length is used.
          std::string t =
              isa<fir::BoxCharType>(ty)
                  ? (annotateUnbounded
                         ? std::string("{[-1]:Pointer, [-1,-1]:Integer}")
                         : std::string("{[-1]:Pointer}"))
                  : procArgType(ty, dl);
          if (!t.empty() && !fn.getArgAttr(i, kTypeAttr))
            fn.setArgAttr(i, kTypeAttr, StringAttr::get(ctx, t));
        }
        if (fty.getNumResults() == 1) {
          std::string t = procArgType(fty.getResult(0), dl);
          if (!t.empty())
            fn->setAttr(kRetTypeAttr, StringAttr::get(ctx, t));
        }
      }

    // Local variables: CHARACTER data (copied with untyped memcpys), and
    // REAL, INTEGER, LOGICAL and COMPLEX scalars and arrays, whose loads and
    // stores flang's TBAA does not type (e.g. an INTEGER count that a callee
    // fills by reference and shift operations then use, MITgcm's
    // ctrl_getobcs*). An alloca is a whole object, so nothing follows the
    // data: a constant size up to Enzyme's type offsets is typed offset by
    // offset, any other at every offset. The conversion of fir.alloca to LLVM
    // keeps only its name, so the function lists its variables by name, with
    // their size for the translation to check.
    if (annotateLocals)
      for (auto fn : module.getOps<func::FuncOp>()) {
        // Names are not unique in a function (e.g. after inlining, ICON's
        // nwp_nh_interface has a CHARACTER and a TYPE(t_wtr_prog) local of
        // the same name): a name with locals of different types is left out.
        llvm::StringMap<std::pair<std::string, int64_t>> types;
        llvm::StringSet<> ambiguous;
        fn.walk([&](fir::AllocaOp alloca) {
          auto name = alloca.getBindcName();
          if (!name)
            return;
          std::optional<std::pair<std::string, int64_t>> t =
              localType(alloca.getInType(), dl);
          auto [it, inserted] = types.try_emplace(
              *name, t ? *t : std::pair<std::string, int64_t>{"", 0});
          if (!t || (!inserted && it->second != *t))
            ambiguous.insert(*name);
        });
        SmallVector<NamedAttribute> locals;
        for (auto &entry : types)
          if (!ambiguous.contains(entry.getKey()))
            locals.push_back(NamedAttribute(
                StringAttr::get(ctx, entry.getKey()),
                ArrayAttr::get(ctx,
                               {StringAttr::get(ctx, entry.getValue().first),
                                IntegerAttr::get(IntegerType::get(ctx, 64),
                                                 entry.getValue().second)})));
        llvm::sort(locals,
                   [](const NamedAttribute &a, const NamedAttribute &b) {
                     return a.getName().strref() < b.getName().strref();
                   });
        if (!locals.empty())
          fn->setAttr(kLocalTypesAttr, DictionaryAttr::get(ctx, locals));
      }

    // Character literals: character data throughout.
    for (fir::GlobalOp global : module.getOps<fir::GlobalOp>())
      if (annotateLiterals && isa<fir::CharacterType>(global.getType()) &&
          !global->hasAttr(kTypeAttr))
        global->setAttr(
            kTypeAttr, StringAttr::get(ctx, "{[-1]:Pointer, [-1,-1]:Integer}"));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::enzyme::createFIRTypeAnnotationsPass() {
  return std::make_unique<FIRTypeAnnotationsPass>();
}

void mlir::enzyme::registerFIRTypeAnnotationsPass() {
  PassRegistration<FIRTypeAnnotationsPass>();
}
