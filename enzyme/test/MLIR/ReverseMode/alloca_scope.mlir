// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func private @scope_overwrite(%x: memref<f32>) {
  memref.alloca_scope {
    %val = memref.load %x[] : memref<f32>
    %cos = math.cos %val : f32
    memref.store %cos, %x[] : memref<f32>
  }
  return
}

func.func @dscope_overwrite(%x: memref<f32>, %dx: memref<f32>) {
  enzyme.autodiff @scope_overwrite(%x, %dx) {
    activity = [#enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (memref<f32>, memref<f32>) -> ()
  return
}

func.func private @scope_load_mul(%x: memref<f64>, %c: f64) -> f64 {
  %r = memref.alloca_scope -> (f64) {
    %v = memref.load %x[] : memref<f64>
    %p = arith.mulf %v, %c : f64
    memref.alloca_scope.return %p : f64
  }
  return %r : f64
}

func.func @dscope_load_mul(%x: memref<f64>, %dx: memref<f64>,
                           %c: f64, %dout: f64) -> f64 {
  %res = enzyme.autodiff @scope_load_mul(%x, %dx, %c, %dout) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (memref<f64>, memref<f64>, f64, f64) -> f64
  return %res : f64
}

// CHECK-LABEL:   func.func private @diffescope_overwrite(
// CHECK-SAME:      %[[X:.*]]: memref<f32>,
// CHECK-SAME:      %[[DX:.*]]: memref<f32>) {
// CHECK:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CDX0:.*]] = "enzyme.init"() : () -> !enzyme.Cache<memref<f32>>
// CHECK:           %[[CVAL:.*]] = "enzyme.init"() : () -> !enzyme.Cache<f32>
// CHECK:           %[[CDX1:.*]] = "enzyme.init"() : () -> !enzyme.Cache<memref<f32>>
// CHECK:           memref.alloca_scope  {
// CHECK:             "enzyme.push"(%[[CDX0]], %[[DX]])
// CHECK:             %[[V:.*]] = memref.load %[[X]][] : memref<f32>
// CHECK:             "enzyme.push"(%[[CVAL]], %[[V]])
// CHECK:             %[[COS:.*]] = math.cos %[[V]] : f32
// CHECK:             "enzyme.push"(%[[CDX1]], %[[DX]])
// CHECK:             memref.store %[[COS]], %[[X]][] : memref<f32>
// CHECK:           }
// CHECK:           %[[DX_R1:.*]] = "enzyme.pop"(%[[CDX1]])
// CHECK:           %[[DOUT:.*]] = memref.load %[[DX_R1]][] : memref<f32>
// CHECK:           memref.store %[[ZERO]], %[[DX_R1]][] : memref<f32>
// CHECK:           %[[VAL:.*]] = "enzyme.pop"(%[[CVAL]])
// CHECK:           %[[SIN:.*]] = math.sin %[[VAL]] : f32
// CHECK:           %[[NEG:.*]] = arith.negf %[[SIN]] : f32
// CHECK:           %[[GRAD:.*]] = arith.mulf %[[DOUT]], %[[NEG]] : f32
// CHECK:           %[[DX_R0:.*]] = "enzyme.pop"(%[[CDX0]])
// CHECK:           %[[OLD:.*]] = memref.load %[[DX_R0]][] : memref<f32>
// CHECK:           %[[NEW:.*]] = arith.addf %[[OLD]], %[[GRAD]] : f32
// CHECK:           memref.store %[[NEW]], %[[DX_R0]][] : memref<f32>
// CHECK:           return

// CHECK-LABEL:   func.func private @diffescope_load_mul(
// CHECK-SAME:      %[[X:[^,]+]]: memref<f64>,
// CHECK-SAME:      %[[DX:[^,]+]]: memref<f64>,
// CHECK-SAME:      %[[C:[^,]+]]: f64,
// CHECK-SAME:      %[[DOUT:[^)]+]]: f64) -> f64 {
// CHECK:           %[[CDX:.*]] = "enzyme.init"() : () -> !enzyme.Cache<memref<f64>>
// CHECK:           %[[GC:.*]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK:           %[[CC:.*]] = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK:           %[[CV:.*]] = "enzyme.init"() : () -> !enzyme.Cache<f64>
// CHECK:           memref.alloca_scope  -> (f64) {
// CHECK:             "enzyme.push"(%[[CDX]], %[[DX]])
// CHECK:             %[[V:.*]] = memref.load %[[X]][] : memref<f64>
// CHECK:             "enzyme.push"(%[[CV]], %[[V]])
// CHECK:             "enzyme.push"(%[[CC]], %[[C]])
// CHECK:             %[[P:.*]] = arith.mulf %[[V]], %[[C]] : f64
// CHECK:             memref.alloca_scope.return %[[P]] : f64
// CHECK:           }
// CHECK:           memref.alloca_scope  {
// CHECK:             %[[V_R:.*]] = "enzyme.pop"(%[[CV]])
// CHECK:             %[[C_R:.*]] = "enzyme.pop"(%[[CC]])
// CHECK-DAG:         %[[DC:.*]] = arith.mulf %[[DOUT]], %[[V_R]] : f64
// CHECK-DAG:         %[[DV:.*]] = arith.mulf %[[DOUT]], %[[C_R]] : f64
// CHECK:             %[[DX_R:.*]] = "enzyme.pop"(%[[CDX]])
// CHECK:             %[[OLD:.*]] = memref.load %[[DX_R]][] : memref<f64>
// CHECK:             %[[NEW:.*]] = arith.addf %[[OLD]], %[[DV]] : f64
// CHECK:             memref.store %[[NEW]], %[[DX_R]][] : memref<f64>
// CHECK:           }
// CHECK:           %[[OUT:.*]] = "enzyme.get"(%[[GC]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           return %[[OUT]] : f64
