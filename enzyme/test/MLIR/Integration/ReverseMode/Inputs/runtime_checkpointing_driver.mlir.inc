  // Prints, for trip count n: n, the gradient of @f at 0.7 with each
  // checkpointing variant, and 1 if the first matches a central difference.
  func.func @run(%n: index) {
    %x = arith.constant 0.7 : f64
    %one = arith.constant 1.0 : f64
    %eps = arith.constant 1.0e-6 : f64
    %tol = arith.constant 1.0e-6 : f64
    %xp = arith.addf %x, %eps : f64
    %xm = arith.subf %x, %eps : f64
    %fp = call @f(%xp, %n) : (f64, index) -> f64
    %fm = call @f(%xm, %n) : (f64, index) -> f64
    %df = arith.subf %fp, %fm : f64
    %eps2 = arith.addf %eps, %eps : f64
    %fd = arith.divf %df, %eps2 : f64
    %g0 = call @g_bin(%x, %n, %one) : (f64, index, f64) -> f64
    %g1 = call @g_rt_bin2(%x, %n, %one) : (f64, index, f64) -> f64
    %g2 = call @g_rt_bin5(%x, %n, %one) : (f64, index, f64) -> f64
    %g3 = call @g_rt_per3(%x, %n, %one) : (f64, index, f64) -> f64
    %g4 = call @g_rt_per(%x, %n, %one) : (f64, index, f64) -> f64
    %err = arith.subf %fd, %g0 : f64
    %aerr = math.absf %err : f64
    %ag = math.absf %g0 : f64
    %scale = arith.addf %ag, %one : f64
    %bound = arith.mulf %scale, %tol : f64
    %ok = arith.cmpf ole, %aerr, %bound : f64
    %oki = arith.extui %ok : i1 to i64
    %ni = arith.index_cast %n : index to i64
    call @printI64(%ni) : (i64) -> ()
    call @printComma() : () -> ()
    call @printF64(%g0) : (f64) -> ()
    call @printComma() : () -> ()
    call @printF64(%g1) : (f64) -> ()
    call @printComma() : () -> ()
    call @printF64(%g2) : (f64) -> ()
    call @printComma() : () -> ()
    call @printF64(%g3) : (f64) -> ()
    call @printComma() : () -> ()
    call @printF64(%g4) : (f64) -> ()
    call @printComma() : () -> ()
    call @printI64(%oki) : (i64) -> ()
    call @printNewline() : () -> ()
    return
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %c10 = arith.constant 10 : index
    %c37 = arith.constant 37 : index
    call @run(%c0) : (index) -> ()
    call @run(%c1) : (index) -> ()
    call @run(%c2) : (index) -> ()
    call @run(%c7) : (index) -> ()
    call @run(%c10) : (index) -> ()
    call @run(%c37) : (index) -> ()
    return
  }

  func.func private @printI64(i64)
  func.func private @printF64(f64)
  func.func private @printComma()
  func.func private @printNewline()
}
