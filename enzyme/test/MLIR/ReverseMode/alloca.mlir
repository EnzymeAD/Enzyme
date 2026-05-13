// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @foo_flat(%x : f64) -> f64 {
  %buf = memref.alloca() : memref<f64>
  memref.store %x, %buf[] : memref<f64>
  %y = memref.load %buf[] : memref<f64>
  return %y : f64
}

func.func @dfoo_flat(%x: f64, %dout : f64) -> f64 {
  %dx = enzyme.autodiff @foo_flat(%x, %dout) {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f64, f64) -> (f64)
  return %dx : f64
}

// CHECK-LABEL:   func.func private @diffefoo_flat(
// CHECK-SAME:      %[[X:[^,]+]]: f64,
// CHECK-SAME:      %[[DOUT:[^)]+]]: f64) -> f64 {
// A shadow memref.alloca must be created and zero-initialized so the
// reverse-mode adjoint can accumulate gradients into it.
// CHECK-DAG:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[DBUF:.*]] = memref.alloca() : memref<f64>
// CHECK:           memref.store %[[ZERO]], %[[DBUF]][] : memref<f64>
// No leftover placeholders should survive the differentiation.
// CHECK-NOT:       enzyme.placeholder
// The function ultimately returns the gradient w.r.t. x (= dout).
// CHECK:           return %{{.*}} : f64