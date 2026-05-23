// RUN: %eopt --enzyme %s | FileCheck %s

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

// CHECK:           %[[GX:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK:           "enzyme.set"(%[[GX]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[CS:.+]] = "enzyme.init"() : () -> !enzyme.Cache<memref<f64>>
// CHECK:           %[[CL:.+]] = "enzyme.init"() : () -> !enzyme.Cache<memref<f64>>
// CHECK:           %[[GY:.+]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK:           "enzyme.set"(%[[GY]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()

// CHECK:           %[[DBUF:.+]] = memref.alloca() : memref<f64>
// CHECK:           %[[ZERO_INIT:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:           linalg.fill ins(%[[ZERO_INIT]] : f64) outs(%[[DBUF]] : memref<f64>)

// CHECK:           %[[BUF:.+]] = memref.alloca() : memref<f64>
// CHECK:           "enzyme.push"(%[[CS]], %[[DBUF]]) : (!enzyme.Cache<memref<f64>>, memref<f64>) -> ()
// CHECK:           memref.store %[[X]], %[[BUF]][] : memref<f64>
// CHECK:           "enzyme.push"(%[[CL]], %[[DBUF]]) : (!enzyme.Cache<memref<f64>>, memref<f64>) -> ()
// CHECK:           memref.load %[[BUF]][] : memref<f64>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:

// CHECK:           %{{.+}} = "enzyme.get"(%[[GY]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           arith.addf %{{.+}}, %[[DOUT]] : f64
// CHECK:           "enzyme.set"(%[[GY]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()

// CHECK:           %{{.+}} = "enzyme.get"(%[[GY]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           %[[POPL:.+]] = "enzyme.pop"(%[[CL]]) : (!enzyme.Cache<memref<f64>>) -> memref<f64>
// CHECK:           memref.load %[[POPL]][] : memref<f64>
// CHECK:           arith.addf
// CHECK:           memref.store %{{.+}}, %[[POPL]][] : memref<f64>

// CHECK:           %[[POPS:.+]] = "enzyme.pop"(%[[CS]]) : (!enzyme.Cache<memref<f64>>) -> memref<f64>
// CHECK:           memref.load %[[POPS]][] : memref<f64>
// CHECK:           %{{.+}} = "enzyme.get"(%[[GX]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           arith.addf
// CHECK:           "enzyme.set"(%[[GX]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[ZERO_CLEAR:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:           memref.store %[[ZERO_CLEAR]], %[[POPS]][] : memref<f64>

// CHECK:           %{{.+}} = "enzyme.get"(%[[GX]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           return %{{.+}} : f64

// CHECK-NOT:       enzyme.placeholder
