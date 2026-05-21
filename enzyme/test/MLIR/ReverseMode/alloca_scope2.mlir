// RUN: %eopt --enzyme %s | FileCheck %s

func.func @foo(%x : f64) -> f64{
    %out = memref.alloca_scope -> (f64) {
      %buf = memref.alloca() : memref<f64>
      memref.store %x, %buf[] : memref<f64>
      %y = memref.load %buf[] : memref<f64>
      memref.alloca_scope.return %y : f64
    }
    return %out  : f64
}

func.func @dfoo(%x: f64, %dout : f64) -> f64 {
  %dx = enzyme.autodiff @foo(%x, %dout) {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f64, f64) -> (f64)
  return %dx : f64
}

// CHECK-LABEL:   func.func private @diffefoo(
// CHECK-SAME:      %[[X:[^,]+]]: f64,
// CHECK-SAME:      %[[DOUT:[^)]+]]: f64) -> f64 {
// CHECK:           %[[C0:.*]] = "enzyme.init"() : () -> !enzyme.Cache<memref<f64>>
// CHECK:           %[[C1:.*]] = "enzyme.init"() : () -> !enzyme.Cache<memref<f64>>
// CHECK:           %[[G0:.*]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK:           "enzyme.set"(%[[G0]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[G1:.*]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK:           "enzyme.set"(%[[G1]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[G2:.*]] = "enzyme.init"() : () -> !enzyme.Gradient<f64>
// CHECK:           "enzyme.set"(%[[G2]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[T5:.*]] = "enzyme.get"(%[[G2]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           %[[T6:.*]] = arith.addf %[[T5]], %[[DOUT]] : f64
// CHECK:           "enzyme.set"(%[[G2]], %[[T6]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[T7:.*]] = "enzyme.get"(%[[G2]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           "enzyme.set"(%[[G2]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[T8:.*]] = "enzyme.get"(%[[G1]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           "enzyme.set"(%[[G1]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[SCOPE:.*]]:2 = memref.alloca_scope  -> (f64, f64) {
// CHECK:             %[[ALLOCA:.*]] = memref.alloca() : memref<f64>
// CHECK:             memref.store %{{.*}}, %[[ALLOCA]][] : memref<f64>
// CHECK:             %[[ALLOCA5:.*]] = memref.alloca() : memref<f64>
// CHECK:             "enzyme.push"(%[[C0]], %[[ALLOCA]]) : (!enzyme.Cache<memref<f64>>, memref<f64>) -> ()
// CHECK:             memref.store %[[X]], %[[ALLOCA5]][] : memref<f64>
// CHECK:             "enzyme.push"(%[[C1]], %[[ALLOCA]]) : (!enzyme.Cache<memref<f64>>, memref<f64>) -> ()
// CHECK:             %[[LD:.*]] = memref.load %[[ALLOCA5]][] : memref<f64>
// CHECK:             "enzyme.set"(%[[G0]], %{{.*}}) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:             %[[T14:.*]] = "enzyme.get"(%[[G0]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:             %[[T15:.*]] = arith.addf %[[T14]], %[[T7]] : f64
// CHECK:             "enzyme.set"(%[[G0]], %[[T15]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:             %[[T16:.*]] = "enzyme.get"(%[[G0]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:             %[[POP1:.*]] = "enzyme.pop"(%[[C1]]) : (!enzyme.Cache<memref<f64>>) -> memref<f64>
// CHECK:             %[[LD18:.*]] = memref.load %[[POP1]][] : memref<f64>
// CHECK:             %[[T19:.*]] = arith.addf %[[LD18]], %[[T16]] : f64
// CHECK:             memref.store %[[T19]], %[[POP1]][] : memref<f64>
// CHECK:             %[[POP0:.*]] = "enzyme.pop"(%[[C0]]) : (!enzyme.Cache<memref<f64>>) -> memref<f64>
// CHECK:             %[[LD21:.*]] = memref.load %[[POP0]][] : memref<f64>
// CHECK:             %[[T22:.*]] = "enzyme.get"(%[[G1]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:             %[[T23:.*]] = arith.addf %[[T22]], %[[LD21]] : f64
// CHECK:             "enzyme.set"(%[[G1]], %[[T23]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:             memref.store %{{.*}}, %[[POP0]][] : memref<f64>
// CHECK:             %[[T24:.*]] = "enzyme.get"(%[[G1]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:             memref.alloca_scope.return %[[LD]], %[[T24]] : f64, f64
// CHECK:           }
// CHECK:           "enzyme.set"(%[[G1]], %[[T8]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[T10:.*]] = "enzyme.get"(%[[G1]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           %[[T11:.*]] = arith.addf %[[T10]], %[[SCOPE]]#1 : f64
// CHECK:           "enzyme.set"(%[[G1]], %[[T11]]) : (!enzyme.Gradient<f64>, f64) -> ()
// CHECK:           %[[T12:.*]] = "enzyme.get"(%[[G1]]) : (!enzyme.Gradient<f64>) -> f64
// CHECK:           return %[[T12]] : f64
