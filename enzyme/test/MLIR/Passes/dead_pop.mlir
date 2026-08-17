// RUN: %eopt --split-input-file --remove-unnecessary-enzyme-ops %s | FileCheck %s

// Reverse mode caches the branch it took, then finds the adjoint of both arms
// empty. What is left is a cache pushed in one block and popped in another,
// whose popped value nobody reads. PopSimplify cannot pair these -- it gives up
// when the push and the pop are in different blocks -- so the init survived all
// the way to LLVM translation, which has no lowering for it:
//
//   cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
//   registration for dialect for op: enzyme.init
//
// A pop nobody reads is worth keeping only for what it does to the cache, and
// when it is the one pop that cache has there is nothing left to do it for.

module {
  llvm.func @dead(%p: !llvm.ptr, %n: i64) {
    %m1 = arith.constant -1 : i32
    %z = arith.constant 0 : i64
    %c0 = "enzyme.init"() : () -> !enzyme.Cache<i32>
    %c1 = "enzyme.init"() : () -> !enzyme.Cache<i32>
    %cmp = arith.cmpi eq, %n, %z : i64
    "enzyme.push"(%c1, %m1) : (!enzyme.Cache<i32>, i32) -> ()
    "enzyme.push"(%c0, %m1) : (!enzyme.Cache<i32>, i32) -> ()
    cf.cond_br %cmp, ^bb2, ^bb1
  ^bb1:
    llvm.intr.trap
    %0 = "enzyme.pop"(%c0) : (!enzyme.Cache<i32>) -> i32
    cf.br ^bb3
  ^bb2:
    %1 = "enzyme.pop"(%c1) : (!enzyme.Cache<i32>) -> i32
    cf.br ^bb3
  ^bb3:
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @dead
// CHECK-NOT:     enzyme.init
// CHECK-NOT:     enzyme.push
// CHECK-NOT:     enzyme.pop

// -----

// A cache with a second pop is a stack being read: dropping the first would
// hand the second the wrong value.

module {
  llvm.func @live(%p: !llvm.ptr, %n: i64) -> i32 {
    %m1 = arith.constant -1 : i32
    %m2 = arith.constant -2 : i32
    %z = arith.constant 0 : i64
    %c = "enzyme.init"() : () -> !enzyme.Cache<i32>
    %cmp = arith.cmpi eq, %n, %z : i64
    "enzyme.push"(%c, %m1) : (!enzyme.Cache<i32>, i32) -> ()
    "enzyme.push"(%c, %m2) : (!enzyme.Cache<i32>, i32) -> ()
    cf.br ^bb1
  ^bb1:
    %0 = "enzyme.pop"(%c) : (!enzyme.Cache<i32>) -> i32
    %1 = "enzyme.pop"(%c) : (!enzyme.Cache<i32>) -> i32
    llvm.return %1 : i32
  }
}

// CHECK-LABEL: llvm.func @live
// CHECK:         enzyme.init
// CHECK:         "enzyme.pop"
// CHECK:         "enzyme.pop"
