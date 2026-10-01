  // Runs one gradient from fresh memory and prints m[0], m[1], dm[0], dm[1].
  func.func @run(%g: (memref<2xf64>, memref<2xf64>, f64) -> ()) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %x0 = arith.constant 0.4 : f64
    %x1 = arith.constant 1.3 : f64
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    %m = memref.alloc() : memref<2xf64>
    %dm = memref.alloc() : memref<2xf64>
    memref.store %x0, %m[%c0] : memref<2xf64>
    memref.store %x1, %m[%c1] : memref<2xf64>
    memref.store %zero, %dm[%c0] : memref<2xf64>
    memref.store %zero, %dm[%c1] : memref<2xf64>
    func.call_indirect %g(%m, %dm, %one)
        : (memref<2xf64>, memref<2xf64>, f64) -> ()
    %m0 = memref.load %m[%c0] : memref<2xf64>
    %m1 = memref.load %m[%c1] : memref<2xf64>
    %d0 = memref.load %dm[%c0] : memref<2xf64>
    %d1 = memref.load %dm[%c1] : memref<2xf64>
    call @printF64(%m0) : (f64) -> ()
    call @printComma() : () -> ()
    call @printF64(%m1) : (f64) -> ()
    call @printComma() : () -> ()
    call @printF64(%d0) : (f64) -> ()
    call @printComma() : () -> ()
    call @printF64(%d1) : (f64) -> ()
    call @printNewline() : () -> ()
    memref.dealloc %m : memref<2xf64>
    memref.dealloc %dm : memref<2xf64>
    return
  }

  func.func @main() {
    %g_ref = func.constant @g_ref : (memref<2xf64>, memref<2xf64>, f64) -> ()
    call @run(%g_ref) : ((memref<2xf64>, memref<2xf64>, f64) -> ()) -> ()
    %g_bin = func.constant @g_bin : (memref<2xf64>, memref<2xf64>, f64) -> ()
    call @run(%g_bin) : ((memref<2xf64>, memref<2xf64>, f64) -> ()) -> ()
    %g_rt_bin = func.constant @g_rt_bin : (memref<2xf64>, memref<2xf64>, f64) -> ()
    call @run(%g_rt_bin) : ((memref<2xf64>, memref<2xf64>, f64) -> ()) -> ()
    %g_rt_per = func.constant @g_rt_per : (memref<2xf64>, memref<2xf64>, f64) -> ()
    call @run(%g_rt_per) : ((memref<2xf64>, memref<2xf64>, f64) -> ()) -> ()
    return
  }

  func.func private @printF64(f64)
  func.func private @printComma()
  func.func private @printNewline()
}
