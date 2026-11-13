// RUN: %eopt %s \
// RUN:   --enzyme-wrap="infn=scf_nrm2 outfn=dscf_nrm2 argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_active_init outfn=dscf_active_init argTys=enzyme_active,enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_init_used_in_body outfn=dscf_init_used_in_body argTys=enzyme_active,enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_outer_active outfn=dscf_outer_active argTys=enzyme_active,enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_multi_mixed outfn=dscf_multi_mixed argTys=enzyme_active,enzyme_dup retTys=enzyme_active,enzyme_const mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_const_body outfn=dscf_const_body argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_same_val outfn=dscf_same_val argTys=enzyme_dup retTys=enzyme_active,enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_complex outfn=dscf_complex argTys=enzyme_active,enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=scf_nested_in_for outfn=dscf_nested_in_for argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=affine_nrm2 outfn=daffine_nrm2 argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=affine_outer_active outfn=daffine_outer_active argTys=enzyme_active,enzyme_dup,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" \
// RUN:   --enzyme-wrap="infn=affine_mixed outfn=daffine_mixed argTys=enzyme_dup retTys=enzyme_active,enzyme_const mode=ReverseModeCombined" \
// RUN:   --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --flatten-enzyme-caches --canonicalize --cse | FileCheck %s

module {
  func.func @scf_nrm2(%x: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %1 = scf.parallel (%i) = (%c0) to (%dim) step (%c1) init (%cst) -> (f64) {
      %2 = memref.load %x[%i] : memref<?xf64>
      %3 = arith.mulf %2, %2 : f64
      scf.reduce(%3 : f64) {
      ^bb0(%a: f64, %b: f64):
        %s = arith.addf %a, %b : f64
        scf.reduce.return %s : f64
      }
    }
    return %1 : f64
  }

  func.func @scf_active_init(%init: f64, %x: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %1 = scf.parallel (%i) = (%c0) to (%dim) step (%c1) init (%init) -> (f64) {
      %v = memref.load %x[%i] : memref<?xf64>
      %w = arith.mulf %v, %v : f64
      scf.reduce(%w : f64) {
      ^bb0(%a: f64, %b: f64):
        %s = arith.addf %a, %b : f64
        scf.reduce.return %s : f64
      }
    }
    return %1 : f64
  }

  func.func @scf_init_used_in_body(%init: f64, %x: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %1 = scf.parallel (%i) = (%c0) to (%dim) step (%c1) init (%init) -> (f64) {
      %v = memref.load %x[%i] : memref<?xf64>
      %w = arith.mulf %v, %init : f64
      scf.reduce(%w : f64) {
      ^bb0(%a: f64, %b: f64):
        %s = arith.addf %a, %b : f64
        scf.reduce.return %s : f64
      }
    }
    return %1 : f64
  }

  func.func @scf_outer_active(%a: f64, %x: memref<?xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %1 = scf.parallel (%i) = (%c0) to (%dim) step (%c1) init (%cst) -> (f64) {
      %v = memref.load %x[%i] : memref<?xf64>
      %w = arith.mulf %v, %a : f64
      scf.reduce(%w : f64) {
      ^bb0(%p: f64, %q: f64):
        %s = arith.addf %p, %q : f64
        scf.reduce.return %s : f64
      }
    }
    return %1 : f64
  }

  func.func @scf_multi_mixed(%init: f64, %x: memref<?xf64>) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst1 = arith.constant 1.000000e+00 : f64
    %cst2 = arith.constant 2.000000e+00 : f64
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %res0, %res1 = scf.parallel (%i) = (%c0) to (%dim) step (%c1) init (%init, %cst1) -> (f64, f64) {
      %v = memref.load %x[%i] : memref<?xf64>
      scf.reduce(%v, %cst2 : f64, f64) {
      ^bb0(%p: f64, %q: f64):
        %s = arith.addf %p, %q : f64
        scf.reduce.return %s : f64
      }, {
      ^bb0(%p: f64, %q: f64):
        %m = arith.mulf %p, %q : f64
        scf.reduce.return %m : f64
      }
    }
    return %res0, %res1 : f64, f64
  }

  func.func @scf_const_body(%init: f64, %n: index) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.000000e+00 : f64
    %res = scf.parallel (%i) = (%c0) to (%n) step (%c1) init (%init) -> (f64) {
      scf.reduce(%cst : f64) {
      ^bb0(%p: f64, %q: f64):
        %s = arith.addf %p, %q : f64
        scf.reduce.return %s : f64
      }
    }
    return %res : f64
  }

  func.func @scf_same_val(%x: memref<?xf64>) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %dim = memref.dim %x, %c0 : memref<?xf64>
    %r0, %r1 = scf.parallel (%i) = (%c0) to (%dim) step (%c1) init (%cst, %cst) -> (f64, f64) {
      %v = memref.load %x[%i] : memref<?xf64>
      scf.reduce(%v, %v : f64, f64) {
      ^bb0(%p: f64, %q: f64):
        %s = arith.addf %p, %q : f64
        scf.reduce.return %s : f64
      }, {
      ^bb0(%p: f64, %q: f64):
        %s = arith.addf %q, %p : f64
        scf.reduce.return %s : f64
      }
    }
    return %r0, %r1 : f64, f64
  }

  func.func @scf_complex(%init: complex<f64>, %x: memref<4xcomplex<f64>>) -> complex<f64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %res = scf.parallel (%i) = (%c0) to (%c4) step (%c1) init (%init) -> complex<f64> {
      %v = memref.load %x[%i] : memref<4xcomplex<f64>>
      %sq = complex.mul %v, %v : complex<f64>
      scf.reduce(%sq : complex<f64>) {
      ^bb0(%a: complex<f64>, %b: complex<f64>):
        %s = complex.add %a, %b : complex<f64>
        scf.reduce.return %s : complex<f64>
      }
    }
    return %res : complex<f64>
  }

  func.func @scf_nested_in_for(%x: memref<4xf64>) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %outer = scf.for %j = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> f64 {
      %inner = scf.parallel (%i) = (%c0) to (%c4) step (%c1) init (%acc) -> f64 {
        %v = memref.load %x[%i] : memref<4xf64>
        %w = arith.mulf %v, %v : f64
        scf.reduce(%w : f64) {
        ^bb0(%a: f64, %b: f64):
          %s = arith.addf %a, %b : f64
          scf.reduce.return %s : f64
        }
      }
      scf.yield %inner : f64
    }
    return %outer : f64
  }

  func.func @affine_nrm2(%arg0: memref<4xf64>) -> f64 {
    %1 = affine.parallel (%i) = (0) to (4) reduce ("addf") -> (f64) {
      %2 = affine.load %arg0[%i] : memref<4xf64>
      %3 = arith.mulf %2, %2 : f64
      affine.yield %3 : f64
    }
    return %1 : f64
  }

  func.func @affine_outer_active(%a: f64, %x: memref<?xf64>, %n: index) -> f64 {
    %1 = affine.parallel (%i) = (0) to (%n) reduce ("addf") -> (f64) {
      %v = affine.load %x[%i] : memref<?xf64>
      %w = arith.mulf %v, %a : f64
      affine.yield %w : f64
    }
    return %1 : f64
  }

  func.func @affine_mixed(%x: memref<4xf64>) -> (f64, i64) {
    %c1_i64 = arith.constant 1 : i64
    %r0, %r1 = affine.parallel (%i) = (0) to (4) reduce ("addf", "addi") -> (f64, i64) {
      %v = affine.load %x[%i] : memref<4xf64>
      affine.yield %v, %c1_i64 : f64, i64
    }
    return %r0, %r1 : f64, i64
  }
}

// CHECK-LABEL: func.func private @dscf_nrm2(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: f64) {
// CHECK:         %[[ALLOC:.*]] = memref.alloc(%[[DIM:.*]]) : memref<?xf64>
// CHECK:         scf.parallel (%[[I:.*]]) = (%{{.*}}) to (%[[DIM]]) step (%{{.*}}) init (%{{.*}}) -> f64 {
// CHECK:           %[[V:.*]] = memref.load %arg0[%[[I]]] : memref<?xf64>
// CHECK:           memref.store %[[V]], %[[ALLOC]][%[[I]]] : memref<?xf64>
// CHECK:           %[[SQ:.*]] = arith.mulf %[[V]], %[[V]] : f64
// CHECK:           scf.reduce(%[[SQ]] : f64) {
// CHECK:             arith.addf
// CHECK:             scf.reduce.return
// CHECK:           }
// CHECK:         }
// CHECK:         scf.parallel (%[[RI:.*]]) = (%{{.*}}) to (%[[DIM]]) step (%{{.*}}) {
// CHECK:           %[[CV:.*]] = memref.load %[[ALLOC]][%[[RI]]] : memref<?xf64>
// CHECK:           %[[M:.*]] = arith.mulf %arg2, %[[CV]] fastmath<fast> : f64
// CHECK:           %[[D:.*]] = arith.addf %[[M]], %[[M]] fastmath<fast> : f64
// CHECK:           enzyme.atomic_rmw addf %[[D]], %arg1[%[[RI]]] monotonic fastmath<fast> : (f64, memref<?xf64>) -> f64
// CHECK:           scf.reduce
// CHECK:         }
// CHECK:         memref.dealloc %[[ALLOC]] : memref<?xf64>
// CHECK:         return
// CHECK:       }

// CHECK-LABEL: func.func private @dscf_active_init(%arg0: f64, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: f64) -> f64 {
// CHECK:         %[[ALLOC:.*]] = memref.alloc(%[[DIM:.*]]) : memref<?xf64>
// CHECK:         scf.parallel
// CHECK:         scf.parallel
// CHECK:           enzyme.atomic_rmw addf %{{.*}}, %arg2[%{{.*}}] monotonic fastmath<fast> : (f64, memref<?xf64>) -> f64
// CHECK:           scf.reduce
// CHECK:         }
// CHECK:         memref.dealloc %[[ALLOC]] : memref<?xf64>
// CHECK:         return %arg3 : f64
// CHECK:       }

// CHECK-LABEL: func.func private @dscf_init_used_in_body(%arg0: f64, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: f64) -> f64 {
// CHECK:         %[[ALLOC:.*]] = memref.alloc(%[[DIM:.*]]) : memref<?xf64>
// CHECK:         scf.parallel
// CHECK:         %[[LOOP_GRAD:.*]] = scf.parallel (%[[RI:.*]]) = (%{{.*}}) to (%[[DIM]]) step (%{{.*}}) init (%{{.*}}) -> f64 {
// CHECK:           %[[CV:.*]] = memref.load %[[ALLOC]][%[[RI]]] : memref<?xf64>
// CHECK:           %[[DX:.*]] = arith.mulf %arg3, %arg0 fastmath<fast> : f64
// CHECK:           %[[DI:.*]] = arith.mulf %arg3, %[[CV]] fastmath<fast> : f64
// CHECK:           enzyme.atomic_rmw addf %[[DX]], %arg2[%[[RI]]] monotonic fastmath<fast> : (f64, memref<?xf64>) -> f64
// CHECK:           scf.reduce(%[[DI]] : f64) {
// CHECK:             arith.addf {{.*}} fastmath<fast> : f64
// CHECK:             scf.reduce.return
// CHECK:           }
// CHECK:         }
// CHECK:         memref.dealloc %[[ALLOC]] : memref<?xf64>
// CHECK:         %[[TOTAL_DI:.*]] = arith.addf %[[LOOP_GRAD]], %arg3 fastmath<fast> : f64
// CHECK:         return %[[TOTAL_DI]] : f64
// CHECK:       }

// CHECK-LABEL: func.func private @dscf_outer_active(%arg0: f64, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: f64) -> f64 {
// CHECK:         %[[ALLOC:.*]] = memref.alloc(%[[DIM:.*]]) : memref<?xf64>
// CHECK:         scf.parallel
// CHECK:         %[[DA:.*]] = scf.parallel (%[[RI:.*]]) = (%{{.*}}) to (%[[DIM]]) step (%{{.*}}) init (%{{.*}}) -> f64 {
// CHECK:           %[[CV:.*]] = memref.load %[[ALLOC]][%[[RI]]] : memref<?xf64>
// CHECK:           %[[DX:.*]] = arith.mulf %arg3, %arg0 fastmath<fast> : f64
// CHECK:           %[[TERM:.*]] = arith.mulf %arg3, %[[CV]] fastmath<fast> : f64
// CHECK:           enzyme.atomic_rmw addf %[[DX]], %arg2[%[[RI]]] monotonic fastmath<fast> : (f64, memref<?xf64>) -> f64
// CHECK:           scf.reduce(%[[TERM]] : f64) {
// CHECK:             arith.addf {{.*}} fastmath<fast> : f64
// CHECK:             scf.reduce.return
// CHECK:           }
// CHECK:         }
// CHECK:         memref.dealloc %[[ALLOC]] : memref<?xf64>
// CHECK:         return %[[DA]] : f64
// CHECK:       }

// CHECK-LABEL: func.func private @dscf_multi_mixed(%arg0: f64, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: f64) -> f64 {
// CHECK:         scf.parallel
// CHECK:           enzyme.atomic_rmw addf %arg3, %arg2[%{{.*}}] monotonic fastmath<fast> : (f64, memref<?xf64>) -> f64
// CHECK:           scf.reduce
// CHECK:         }
// CHECK:         return %arg3 : f64
// CHECK:       }

// CHECK-LABEL: func.func private @dscf_const_body(%arg0: f64, %arg1: index, %arg2: f64) -> f64 {
// CHECK-NEXT:    return %arg2 : f64
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @dscf_same_val(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: f64, %arg3: f64) {
// CHECK:         %[[SUM:.*]] = arith.addf %arg2, %arg3 fastmath<fast> : f64
// CHECK:         scf.parallel
// CHECK:           enzyme.atomic_rmw addf %[[SUM]], %arg1[%{{.*}}] monotonic fastmath<fast> : (f64, memref<?xf64>) -> f64
// CHECK:           scf.reduce
// CHECK:         }
// CHECK:         return
// CHECK:       }

// CHECK-LABEL: func.func private @dscf_complex(%arg0: complex<f64>, %arg1: memref<4xcomplex<f64>>, %arg2: memref<4xcomplex<f64>>, %arg3: complex<f64>) -> complex<f64> {
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<4xcomplex<f64>>
// CHECK:         scf.parallel
// CHECK:           complex.mul
// CHECK:           scf.reduce
// CHECK:             complex.add
// CHECK:             scf.reduce.return
// CHECK:         scf.parallel
// CHECK:           enzyme.atomic_rmw addf %{{.*}}, %arg2[%{{.*}}] monotonic fastmath<fast> : (complex<f64>, memref<4xcomplex<f64>>) -> complex<f64>
// CHECK:           scf.reduce
// CHECK:         }
// CHECK:         memref.dealloc %[[ALLOC]] : memref<4xcomplex<f64>>
// CHECK:         return %arg3 : complex<f64>
// CHECK:       }

// CHECK-LABEL: func.func private @dscf_nested_in_for(%arg0: memref<4xf64>, %arg1: memref<4xf64>, %arg2: f64) {
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<4x4xf64>
// CHECK:         scf.for
// CHECK:           scf.parallel
// CHECK:             memref.store
// CHECK:             scf.reduce
// CHECK:         scf.for
// CHECK:           scf.parallel
// CHECK:             memref.load
// CHECK:             enzyme.atomic_rmw addf %{{.*}}, %arg1[%{{.*}}] monotonic fastmath<fast> : (f64, memref<4xf64>) -> f64
// CHECK:             scf.reduce
// CHECK:         memref.dealloc %[[ALLOC]] : memref<4x4xf64>
// CHECK:         return
// CHECK:       }

// CHECK-LABEL: func.func private @daffine_nrm2(%arg0: memref<4xf64>, %arg1: memref<4xf64>, %arg2: f64) {
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<4xf64>
// CHECK:         affine.parallel (%[[I:.*]]) = (0) to (4) reduce ("addf") -> (f64) {
// CHECK:           %[[V:.*]] = affine.load %arg0[%[[I]]] : memref<4xf64>
// CHECK:           memref.store %[[V]], %[[ALLOC]][%[[I]]] : memref<4xf64>
// CHECK:           %[[SQ:.*]] = arith.mulf %[[V]], %[[V]] : f64
// CHECK:           affine.yield %[[SQ]] : f64
// CHECK:         }
// CHECK:         affine.parallel (%[[RI:.*]]) = (0) to (4) {
// CHECK:           %[[CV:.*]] = memref.load %[[ALLOC]][%[[RI]]] : memref<4xf64>
// CHECK:           %[[M:.*]] = arith.mulf %arg2, %[[CV]] fastmath<fast> : f64
// CHECK:           %[[D:.*]] = arith.addf %[[M]], %[[M]] fastmath<fast> : f64
// CHECK:           enzyme.atomic_rmw addf %[[D]], %arg1[%[[RI]]] monotonic fastmath<fast> : (f64, memref<4xf64>) -> f64
// CHECK:         }
// CHECK:         memref.dealloc %[[ALLOC]] : memref<4xf64>
// CHECK:         return
// CHECK:       }

// CHECK-LABEL: func.func private @daffine_outer_active(%arg0: f64, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: index, %arg4: f64) -> f64 {
// CHECK:         %[[ALLOC:.*]] = memref.alloc(%arg3) : memref<?xf64>
// CHECK:         affine.parallel (%{{.*}}) = (0) to (symbol(%arg3)) reduce ("addf") -> (f64) {
// CHECK:         %[[DA:.*]] = affine.parallel (%[[RI:.*]]) = (0) to (symbol(%arg3)) reduce ("addf") -> (f64) {
// CHECK:           %[[CV:.*]] = memref.load %[[ALLOC]][%[[RI]]] : memref<?xf64>
// CHECK:           %[[DX:.*]] = arith.mulf %arg4, %arg0 fastmath<fast> : f64
// CHECK:           %[[TERM:.*]] = arith.mulf %arg4, %[[CV]] fastmath<fast> : f64
// CHECK:           enzyme.atomic_rmw addf %[[DX]], %arg2[%[[RI]]] monotonic fastmath<fast> : (f64, memref<?xf64>) -> f64
// CHECK:           affine.yield %[[TERM]] : f64
// CHECK:         }
// CHECK:         memref.dealloc %[[ALLOC]] : memref<?xf64>
// CHECK:         return %[[DA]] : f64
// CHECK:       }

// CHECK-LABEL: func.func private @daffine_mixed(%arg0: memref<4xf64>, %arg1: memref<4xf64>, %arg2: f64) {
// CHECK:         affine.parallel (%[[RI:.*]]) = (0) to (4) {
// CHECK:           enzyme.atomic_rmw addf %arg2, %arg1[%[[RI]]] monotonic fastmath<fast> : (f64, memref<4xf64>) -> f64
// CHECK:         }
// CHECK:         return
// CHECK:       }
