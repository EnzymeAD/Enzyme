// RUN: %eopt --inline-enzyme-regions %s | FileCheck %s

// A function with no enzyme regions must pass through untouched: the
// driver's region simplification must not merge structurally identical
// blocks that differ only in constants -- LLVM cannot split such shared
// tails apart again, and its backend hoists the merged tail's setup into
// the hot path.

llvm.func @use(i64)
llvm.func @twin_tails(%c: i1) {
  llvm.cond_br %c, ^bb1, ^bb2
^bb1:
  %a = llvm.mlir.constant(27 : i64) : i64
  llvm.call @use(%a) : (i64) -> ()
  llvm.br ^bb3
^bb2:
  %b = llvm.mlir.constant(20 : i64) : i64
  llvm.call @use(%b) : (i64) -> ()
  llvm.br ^bb3
^bb3:
  llvm.return
}

// CHECK-LABEL: llvm.func @twin_tails
// CHECK: ^bb1:
// CHECK-NEXT: llvm.call @use
// CHECK: ^bb2:
// CHECK-NEXT: llvm.call @use
// CHECK: ^bb3:
