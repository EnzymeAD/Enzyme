// RUN: %etypetreetest | FileCheck %s
// RUN: %etypetreetest | %etranslate -enzyme-to-llvmir | FileCheck %s --check-prefix=LLVMIR

//===- type_tree_attr.cpp - Test the conversion of MTypeTree --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds type trees with MTypeTree, converts them with getTypeTreeAttr and
// prints a module that carries them, for FileCheck.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "Interfaces/EnzymeLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::enzyme;

/// Adds a global of 16 bytes named `name`, with the type tree of `tree` if it
/// has one.
static void addGlobal(OpBuilder &builder, StringRef name,
                      const MTypeTree &tree) {
  MLIRContext *context = builder.getContext();
  auto global = LLVM::GlobalOp::create(
      builder, builder.getUnknownLoc(),
      LLVM::LLVMArrayType::get(IntegerType::get(context, 8), 16),
      /*isConstant=*/false, LLVM::Linkage::External, name, Attribute());
  if (TypeTreeAttr attr = getTypeTreeAttr(context, tree))
    global->setAttr(EnzymeDialect::getTypeTreeAttrName(), attr);
}

// CHECK-LABEL: llvm.mlir.global external @scalar()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: f32}>
// LLVMIR-DAG: @scalar = external global [16 x i8], !enzyme_type ![[SCALAR:[0-9]+]]
// LLVMIR-DAG: ![[SCALAR]] = !{!"Unknown", i32 -1, ![[FLOAT:[0-9]+]]}
// LLVMIR-DAG: ![[FLOAT]] = !{!"Float@float"}

static MTypeTree scalar(MLIRContext *context) {
  return MTypeTree(MConcreteType(Float32Type::get(context))).Only(-1, nullptr);
}

// CHECK-LABEL: llvm.mlir.global external @mixed()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: pointer {0: f64, 8: integer, 16: anything}}>

static MTypeTree mixed(MLIRContext *context) {
  MTypeTree tree;
  tree.insert({-1}, BaseType::Pointer);
  tree.insert({-1, 0}, Float64Type::get(context));
  tree.insert({-1, 8}, BaseType::Integer);
  tree.insert({-1, 16}, BaseType::Anything);
  return tree;
}

// CHECK-LABEL: llvm.mlir.global external @nested()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{-1: pointer {0: pointer {-1: bf16}, 8: f64}}>
// LLVMIR-DAG: @nested = external global [16 x i8], !enzyme_type ![[NESTED:[0-9]+]]
// LLVMIR-DAG: ![[NESTED]] = !{!"Unknown", i32 -1, ![[NESTEDP:[0-9]+]]}
// LLVMIR-DAG: ![[NESTEDP]] = !{!"Pointer", i32 0, ![[INNERP:[0-9]+]], i32 8, ![[DOUBLE:[0-9]+]]}
// LLVMIR-DAG: ![[INNERP]] = !{!"Pointer", i32 -1, ![[BF16:[0-9]+]]}
// LLVMIR-DAG: ![[BF16]] = !{!"Float@bf16"}
// LLVMIR-DAG: ![[DOUBLE]] = !{!"Float@double"}

static MTypeTree nested(MLIRContext *context) {
  MTypeTree tree;
  tree.insert({-1}, BaseType::Pointer);
  tree.insert({-1, 0}, BaseType::Pointer);
  tree.insert({-1, 0, -1}, BFloat16Type::get(context));
  tree.insert({-1, 8}, Float64Type::get(context));
  return tree;
}

// CHECK-LABEL: llvm.mlir.global external @root()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<pointer {0: integer}>

/// A tree that types the value itself.
static MTypeTree root() {
  MTypeTree tree(BaseType::Pointer);
  tree.insert({0}, BaseType::Integer);
  return tree;
}

// CHECK-LABEL: llvm.mlir.global external @unknown()
// CHECK-SAME: enzyme.type = #enzyme.type_tree<{8: integer}>

/// A tree with an entry of unknown type, which the attribute leaves out.
static MTypeTree unknown() {
  MTypeTree tree;
  tree.insert({0}, BaseType::Unknown);
  tree.insert({8}, BaseType::Integer);
  return tree;
}

// An empty tree has no attribute.
// CHECK-LABEL: llvm.mlir.global external @empty()
// CHECK-NOT: enzyme.type
// CHECK-SAME: !llvm.array<16 x i8>

// CHECK-LABEL: llvm.func @f(!llvm.ptr {enzyme.type = #enzyme.type_tree<{-1: pointer {0: pointer {-1: bf16}, 8: f64}}>})
// LLVMIR-DAG: declare void @f(ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,0,-1]:Float@bf16, [-1,8]:Float@double}")

int main() {
  MLIRContext context;
  context.loadDialect<EnzymeDialect, LLVM::LLVMDialect>();
  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module->getBody());

  addGlobal(builder, "scalar", scalar(&context));
  addGlobal(builder, "mixed", mixed(&context));
  addGlobal(builder, "nested", nested(&context));
  addGlobal(builder, "root", root());
  addGlobal(builder, "unknown", unknown());
  addGlobal(builder, "empty", MTypeTree());

  auto function = LLVM::LLVMFuncOp::create(
      builder, builder.getUnknownLoc(), "f",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&context),
                                  {LLVM::LLVMPointerType::get(&context)}));
  function.setArgAttr(0, EnzymeDialect::getTypeTreeAttrName(),
                      getTypeTreeAttr(&context, nested(&context)));

  module->print(llvm::outs());
  return 0;
}
