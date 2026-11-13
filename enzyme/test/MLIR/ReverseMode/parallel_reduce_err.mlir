// RUN: %eopt %s --split-input-file --verify-diagnostics --enzyme

module {
  func.func @scf_mul(%x: memref<4xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.0 : f64
    // expected-error@+1 {{reverse mode of non-additive scf.parallel reductions is not yet implemented}}
    %res = scf.parallel (%i) = (%c0) to (%c4) step (%c1) init (%cst) -> f64 {
      %v = memref.load %x[%i] : memref<4xf64>
      scf.reduce(%v : f64) {
      ^bb0(%a: f64, %b: f64):
        %m = arith.mulf %a, %b : f64
        scf.reduce.return %m : f64
      }
    }
    return %res : f64
  }

  func.func @dscf_mul(%x: memref<4xf64>, %dx: memref<4xf64>, %dret: f64) -> f64 {
    %r = enzyme.autodiff @scf_mul(%x, %dx, %dret) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (memref<4xf64>, memref<4xf64>, f64) -> f64
    return %r : f64
  }
}

// -----

module {
  func.func @affine_mul(%x: memref<4xf64>) -> f64 {
    // expected-error@+1 {{reverse mode of active mulf affine.parallel reduction is not yet implemented}}
    %res = affine.parallel (%i) = (0) to (4) reduce ("mulf") -> (f64) {
      %v = affine.load %x[%i] : memref<4xf64>
      affine.yield %v : f64
    }
    return %res : f64
  }

  func.func @daffine_mul(%x: memref<4xf64>, %dx: memref<4xf64>, %dret: f64) -> f64 {
    %r = enzyme.autodiff @affine_mul(%x, %dx, %dret) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (memref<4xf64>, memref<4xf64>, f64) -> f64
    return %r : f64
  }
}
