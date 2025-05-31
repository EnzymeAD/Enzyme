// RUN: %eopt --probprog %s | FileCheck %s

module {
  func.func private @normal(%rng : memref<i64>, %mean : f64, %stddev : f64) -> f64

  func.func @test(%rng : memref<i64>, %mean : f64, %stddev : f64) -> f64 {
    %s_sym = llvm.mlir.constant(1 : i64) : i64
    %t_sym = llvm.mlir.constant(2 : i64) : i64
    %s = enzyme.sample [%s_sym : i64] @normal(%rng, %mean, %stddev) { name="s" } : (memref<i64>, f64, f64) -> f64
    %t = enzyme.sample [%t_sym : i64] @normal(%rng, %s, %stddev) { name="t" } : (memref<i64>, f64, f64) -> f64
    return %t : f64
  }

  func.func @foo(%rng : memref<i64>, %x : f64, %y : f64) -> f64 {
    %res = enzyme.generate @test(%rng, %x, %x) { name="res" } : (memref<i64>, f64, f64) -> f64
    return %res : f64
  }
}

// CHECK: func.func @test.generate(%[[rng:.+]]: memref<i64>, %[[mean:.+]]: f64, %[[stddev:.+]]: f64) -> f64 {
// CHECK-NEXT: %[[s_sym:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[t_sym:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[s:.+]] = call @normal(%[[rng]], %[[mean]], %[[stddev]]) : (memref<i64>, f64, f64) -> f64
// CHECK-NEXT: %[[t:.+]] = call @normal(%[[rng]], %[[s]], %[[stddev]]) : (memref<i64>, f64, f64) -> f64
// CHECK-NEXT: return %[[t]] : f64
// CHECK-NEXT: }
