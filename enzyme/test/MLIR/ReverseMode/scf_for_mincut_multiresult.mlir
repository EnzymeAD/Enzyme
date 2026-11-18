// RUN: %eopt %s --pass-pipeline="builtin.module(enzyme{dataflow markReadonly},canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math)" --split-input-file | FileCheck %s

// Multi-result scf.if containing inner enzyme.push ops (from preserve_cache)
// where both results can be recomputed from already-cached inputs (%cond, %a,
// %b). Only those 3 inputs should be cached (not %res#0 or %res#1).
module {
  func.func private @multi_if_recompute_both(%cond: memref<?xi1>, %a: memref<?xf64>, %b: memref<?xf64>, %x: f64) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %r:2 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (f64, f64) {
      %c = memref.load %cond[%iv] : memref<?xi1>
      %av = memref.load %a[%iv] : memref<?xf64>
      %bv = memref.load %b[%iv] : memref<?xf64>
      %res:2 = scf.if %c -> (f64, f64) {
        %p0 = math.cos %x : f64
        %prod = arith.mulf %av, %bv : f64
        %p1 = math.sin %prod : f64
        scf.yield %p0, %p1 : f64, f64
      } else {
        %q0 = math.sin %x : f64
        %sum = arith.addf %av, %bv : f64
        %q1 = math.tanh %sum : f64
        scf.yield %q0, %q1 : f64, f64
      }
      %s0 = arith.mulf %res#0, %res#0 : f64
      %s1 = arith.mulf %res#1, %res#1 : f64
      %na0 = arith.addf %acc0, %s0 : f64
      %na1 = arith.addf %acc1, %s1 : f64
      scf.yield %na0, %na1 : f64, f64
    }
    memref.store %zero, %a[%c0] : memref<?xf64>
    memref.store %zero, %b[%c0] : memref<?xf64>
    return %r#0, %r#1 : f64, f64
  }

  func.func @dmulti_if_recompute_both(%cond: memref<?xi1>, %a: memref<?xf64>, %da: memref<?xf64>, %b: memref<?xf64>, %db: memref<?xf64>, %x: f64, %dr0: f64, %dr1: f64) -> f64 {
    %dx = enzyme.autodiff @multi_if_recompute_both(%cond, %a, %da, %b, %db, %x, %dr0, %dr1) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_activenoneed>]
    }> : (memref<?xi1>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, f64, f64, f64) -> f64
    return %dx : f64
  }
}

// CHECK-LABEL: func.func private @diffemulti_if_recompute_both(
// CHECK:         %[[ALLOC_C:.*]] = memref.alloc() : memref<10xi1>
// CHECK:         %[[ALLOC_A:.*]] = memref.alloc() : memref<10xf64>
// CHECK:         %[[ALLOC_B:.*]] = memref.alloc() : memref<10xf64>
// CHECK-NOT:     memref.alloc
// CHECK:         scf.for
// CHECK:           scf.if {{.*}} -> (f64, f64)
// CHECK:         scf.for
// CHECK:           %[[REV_C:.*]] = memref.load %[[ALLOC_C]]
// CHECK:           %[[REV_A:.*]] = memref.load %[[ALLOC_A]]
// CHECK:           %[[REV_B:.*]] = memref.load %[[ALLOC_B]]
// CHECK:           %[[RECOMP:.*]]:2 = scf.if %[[REV_C]] -> (f64, f64) {
// CHECK:             math.cos
// CHECK:             arith.mulf %[[REV_A]], %[[REV_B]]
// CHECK:             math.sin
// CHECK:             scf.yield
// CHECK:           } else {
// CHECK:             math.sin
// CHECK:             arith.addf %[[REV_A]], %[[REV_B]]
// CHECK:             math.tanh
// CHECK:             scf.yield
// CHECK:           }

// -----

// Multi-result scf.if where res#0 depends only on the already-cached condition
// and a loop-invariant argument (%x), while res#1 depends on two uncached
// non-readonly loads (%a, %b) whose adjoints do not need %a/%b. Mincut caches
// only (%cond, res#1) and emits a 1-result scf.if in reverse to recompute
// res#0.
module {
  func.func private @multi_if_partial_mincut(%cond: memref<?xi1>, %a: memref<?xf64>, %b: memref<?xf64>, %x: f64) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %r:2 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (f64, f64) {
      %c = memref.load %cond[%iv] : memref<?xi1>
      %av = memref.load %a[%iv] : memref<?xf64>
      %bv = memref.load %b[%iv] : memref<?xf64>
      %res:2 = scf.if %c -> (f64, f64) {
        %p0 = math.cos %x : f64
        %p1 = arith.addf %av, %bv : f64
        scf.yield %p0, %p1 : f64, f64
      } else {
        %q0 = math.sin %x : f64
        %q1 = arith.subf %av, %bv : f64
        scf.yield %q0, %q1 : f64, f64
      }
      %s0 = arith.mulf %res#0, %res#0 : f64
      %s1 = arith.mulf %res#1, %res#1 : f64
      %na0 = arith.addf %acc0, %s0 : f64
      %na1 = arith.addf %acc1, %s1 : f64
      scf.yield %na0, %na1 : f64, f64
    }
    memref.store %zero, %a[%c0] : memref<?xf64>
    memref.store %zero, %b[%c0] : memref<?xf64>
    return %r#0, %r#1 : f64, f64
  }

  func.func @dmulti_if_partial_mincut(%cond: memref<?xi1>, %a: memref<?xf64>, %da: memref<?xf64>, %b: memref<?xf64>, %db: memref<?xf64>, %x: f64, %dr0: f64, %dr1: f64) -> f64 {
    %dx = enzyme.autodiff @multi_if_partial_mincut(%cond, %a, %da, %b, %db, %x, %dr0, %dr1) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_activenoneed>]
    }> : (memref<?xi1>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, f64, f64, f64) -> f64
    return %dx : f64
  }
}

// CHECK-LABEL: func.func private @diffemulti_if_partial_mincut(
// CHECK:         %[[ALLOC_C:.*]] = memref.alloc() : memref<10xi1>
// CHECK:         %[[ALLOC_R1:.*]] = memref.alloc() : memref<10xf64>
// CHECK-NOT:     memref.alloc
// CHECK:         scf.for
// CHECK:           %[[FWD_IF:.*]]:2 = scf.if {{.*}} -> (f64, f64)
// CHECK:           memref.store %[[FWD_IF]]#1, %[[ALLOC_R1]]
// CHECK:         scf.for
// CHECK:           %[[REV_C:.*]] = memref.load %[[ALLOC_C]]
// CHECK:           %[[REV_R1:.*]] = memref.load %[[ALLOC_R1]]
// CHECK:           %[[REV_R0:.*]] = scf.if %[[REV_C]] -> (f64) {
// CHECK:             %[[COS:.*]] = math.cos
// CHECK-NEXT:        scf.yield %[[COS]] : f64
// CHECK:           } else {
// CHECK:             %[[SIN:.*]] = math.sin
// CHECK-NEXT:        scf.yield %[[SIN]] : f64
// CHECK:           }

// -----

// Multi-result scf.if with an unmovable non-readonly load feeding res#1 and an
// unrelated memref.store side effect inside the then region. Only res#1 is
// cached (along with %cond), while res#0 is sliced into a pure 1-result scf.if
// in reverse without cloning the store or unmovable load.
module {
  func.func private @multi_if_side_effect(%cond: memref<?xi1>, %a: memref<?xf64>, %scratch: memref<?xf64>, %x: f64) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %r:2 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (f64, f64) {
      %c = memref.load %cond[%iv] : memref<?xi1>
      %res:2 = scf.if %c -> (f64, f64) {
        %p0 = math.cos %x : f64
        memref.store %p0, %scratch[%iv] : memref<?xf64>
        %p1 = memref.load %a[%iv] : memref<?xf64>
        scf.yield %p0, %p1 : f64, f64
      } else {
        %q0 = math.sin %x : f64
        %q1 = memref.load %a[%iv] : memref<?xf64>
        scf.yield %q0, %q1 : f64, f64
      }
      %s0 = arith.mulf %res#0, %res#0 : f64
      %s1 = arith.mulf %res#1, %res#1 : f64
      %na0 = arith.addf %acc0, %s0 : f64
      %na1 = arith.addf %acc1, %s1 : f64
      scf.yield %na0, %na1 : f64, f64
    }
    memref.store %zero, %a[%c0] : memref<?xf64>
    return %r#0, %r#1 : f64, f64
  }

  func.func @dmulti_if_side_effect(%cond: memref<?xi1>, %a: memref<?xf64>, %da: memref<?xf64>, %scratch: memref<?xf64>, %x: f64, %dr0: f64, %dr1: f64) -> f64 {
    %dx = enzyme.autodiff @multi_if_side_effect(%cond, %a, %da, %scratch, %x, %dr0, %dr1) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_activenoneed>]
    }> : (memref<?xi1>, memref<?xf64>, memref<?xf64>, memref<?xf64>, f64, f64, f64) -> f64
    return %dx : f64
  }
}

// CHECK-LABEL: func.func private @diffemulti_if_side_effect(
// CHECK:         %[[ALLOC_C:.*]] = memref.alloc() : memref<10xi1>
// CHECK:         %[[ALLOC_R1:.*]] = memref.alloc() : memref<10xf64>
// CHECK-NOT:     memref.alloc
// CHECK:         scf.for
// CHECK:           %[[FWD_IF:.*]]:2 = scf.if {{.*}} -> (f64, f64)
// CHECK:           memref.store %[[FWD_IF]]#1, %[[ALLOC_R1]]
// CHECK:         scf.for
// CHECK:           %[[REV_C:.*]] = memref.load %[[ALLOC_C]]
// CHECK:           %[[REV_R1:.*]] = memref.load %[[ALLOC_R1]]
// CHECK:           %[[REV_R0:.*]] = scf.if %[[REV_C]] -> (f64) {
// CHECK-NEXT:        %[[COS:.*]] = math.cos
// CHECK-NEXT:        scf.yield %[[COS]] : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[SIN:.*]] = math.sin
// CHECK-NEXT:        scf.yield %[[SIN]] : f64
// CHECK-NEXT:      }

// -----

// Nested multi-result scf.if inside an outer multi-result scf.if where outer
// res#0 needs only inner res#0 (pure), while outer res#1 needs inner res#1
// (unmovable load). Outer res#1 is cached and outer res#0 is recomputed via a
// sliced 1-result outer scf.if containing a sliced 1-result inner scf.if.
module {
  func.func private @nested_multi_if(%c1: memref<?xi1>, %c2: memref<?xi1>, %a: memref<?xf64>, %x: f64) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %zero = arith.constant 0.0 : f64
    %r:2 = scf.for %iv = %c0 to %c10 step %c1_idx iter_args(%acc0 = %zero, %acc1 = %zero) -> (f64, f64) {
      %cond1 = memref.load %c1[%iv] : memref<?xi1>
      %cond2 = memref.load %c2[%iv] : memref<?xi1>
      %outer:2 = scf.if %cond1 -> (f64, f64) {
        %inner:2 = scf.if %cond2 -> (f64, f64) {
          %v0 = math.cos %x : f64
          %v1 = memref.load %a[%iv] : memref<?xf64>
          scf.yield %v0, %v1 : f64, f64
        } else {
          %v0 = math.sin %x : f64
          %v1 = arith.constant 2.0 : f64
          scf.yield %v0, %v1 : f64, f64
        }
        scf.yield %inner#0, %inner#1 : f64, f64
      } else {
        %w0 = math.exp %x : f64
        %w1 = arith.constant 3.0 : f64
        scf.yield %w0, %w1 : f64, f64
      }
      %s0 = arith.mulf %outer#0, %outer#0 : f64
      %s1 = arith.mulf %outer#1, %outer#1 : f64
      %na0 = arith.addf %acc0, %s0 : f64
      %na1 = arith.addf %acc1, %s1 : f64
      scf.yield %na0, %na1 : f64, f64
    }
    memref.store %zero, %a[%c0] : memref<?xf64>
    return %r#0, %r#1 : f64, f64
  }

  func.func @dnested_multi_if(%c1: memref<?xi1>, %c2: memref<?xi1>, %a: memref<?xf64>, %da: memref<?xf64>, %x: f64, %dr0: f64, %dr1: f64) -> f64 {
    %dx = enzyme.autodiff @nested_multi_if(%c1, %c2, %a, %da, %x, %dr0, %dr1) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_activenoneed>]
    }> : (memref<?xi1>, memref<?xi1>, memref<?xf64>, memref<?xf64>, f64, f64, f64) -> f64
    return %dx : f64
  }
}

// CHECK-LABEL: func.func private @diffenested_multi_if(
// CHECK:         %[[ALLOC_C1:.*]] = memref.alloc() : memref<10xi1>
// CHECK:         %[[ALLOC_C2:.*]] = memref.alloc() : memref<10xi1>
// CHECK:         %[[ALLOC_R1:.*]] = memref.alloc() : memref<10xf64>
// CHECK-NOT:     memref.alloc
// CHECK:         scf.for
// CHECK:           %[[FWD_OUTER:.*]]:2 = scf.if {{.*}} -> (f64, f64)
// CHECK:           memref.store %[[FWD_OUTER]]#1, %[[ALLOC_R1]]
// CHECK:         scf.for
// CHECK:           %[[REV_C1:.*]] = memref.load %[[ALLOC_C1]]
// CHECK:           %[[REV_C2:.*]] = memref.load %[[ALLOC_C2]]
// CHECK:           %[[REV_R1:.*]] = memref.load %[[ALLOC_R1]]
// CHECK:           scf.if %[[REV_C1]] -> (f64) {
// CHECK-NEXT:        %[[INNER:.*]] = scf.if %[[REV_C2]] -> (f64) {
// CHECK-NEXT:          %[[COS:.*]] = math.cos
// CHECK-NEXT:          scf.yield %[[COS]] : f64
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[SIN:.*]] = math.sin
// CHECK-NEXT:          scf.yield %[[SIN]] : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[INNER]] : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[EXP:.*]] = math.exp
// CHECK-NEXT:        scf.yield %[[EXP]] : f64
// CHECK-NEXT:      }

// -----

// Multi-result affine.if inside affine.for where res#0 is pure from %x and
// res#1 depends on an unmovable load from %a. Only res#1 is cached, and res#0
// is sliced into a 1-result affine.if in reverse.
module {
  func.func private @multi_affine_if(%a: memref<?xf64>, %x: f64) -> (f64, f64) {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f64
    %r:2 = affine.for %iv = 0 to 10 iter_args(%acc0 = %zero, %acc1 = %zero) -> (f64, f64) {
      %res:2 = affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%iv) -> (f64, f64) {
        %p0 = math.cos %x : f64
        %p1 = memref.load %a[%iv] : memref<?xf64>
        affine.yield %p0, %p1 : f64, f64
      } else {
        %q0 = math.sin %x : f64
        %q1 = arith.constant 1.0 : f64
        affine.yield %q0, %q1 : f64, f64
      }
      %s0 = arith.mulf %res#0, %res#0 : f64
      %s1 = arith.mulf %res#1, %res#1 : f64
      %na0 = arith.addf %acc0, %s0 : f64
      %na1 = arith.addf %acc1, %s1 : f64
      affine.yield %na0, %na1 : f64, f64
    }
    memref.store %zero, %a[%c0] : memref<?xf64>
    return %r#0, %r#1 : f64, f64
  }

  func.func @dmulti_affine_if(%a: memref<?xf64>, %da: memref<?xf64>, %x: f64, %dr0: f64, %dr1: f64) -> f64 {
    %dx = enzyme.autodiff @multi_affine_if(%a, %da, %x, %dr0, %dr1) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_activenoneed>]
    }> : (memref<?xf64>, memref<?xf64>, f64, f64, f64) -> f64
    return %dx : f64
  }
}

// CHECK-LABEL: func.func private @diffemulti_affine_if(
// CHECK:         %[[ALLOC_R1:.*]] = memref.alloc() : memref<10xf64>
// CHECK-NOT:     memref.alloc
// CHECK:         affine.for
// CHECK:           %[[FWD_IF:.*]]:2 = affine.if {{.*}} -> (f64, f64)
// CHECK:           memref.store %[[FWD_IF]]#1, %[[ALLOC_R1]]
// CHECK:         affine.for
// CHECK:           %[[REV_R1:.*]] = memref.load %[[ALLOC_R1]]
// CHECK:           %[[REV_R0:.*]] = affine.if {{.*}} -> f64 {
// CHECK-NEXT:        %[[COS:.*]] = math.cos
// CHECK-NEXT:        affine.yield %[[COS]] : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[SIN:.*]] = math.sin
// CHECK-NEXT:        affine.yield %[[SIN]] : f64
// CHECK-NEXT:      }
