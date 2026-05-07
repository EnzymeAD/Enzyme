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
    ret_activity = [#enzyme<acitivity enzyme_activenoneed>]
  } : (f64, f64) -> (f64)
  return %dx : f64
}

func.func @foo2(%x : f64) -> f64{
    %cst = arith.constant 0.0000e+00 : f64
      %buf = memref.alloca() : memref<f64>
      memref.store %x, %buf[] : memref<f64>
      %y = memref.load %buf[] : memref<f64>
      %out = arith.addf %cst, %y : f64
    return %out  : f64
}


func.func @dfoo2(%x: f64, %dout : f64) -> f64 {
  %dx = enzyme.autodiff @foo2(%x, %dout) {
    activity = [#enzyme<activity enzyme_active>],
    ret_activity = [#enzyme<activity enzyme_activenoneed>]
  } : (f64, f64) -> (f64)
  return %dx : f64
}
