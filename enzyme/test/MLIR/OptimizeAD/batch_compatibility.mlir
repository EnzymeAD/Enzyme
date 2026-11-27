// RUN: %eopt --enzyme-diff-batch --split-input-file %s | FileCheck %s
// RUN: %eopt --enzyme-diff-batch --enzyme-diff-batch --split-input-file %s | FileCheck %s

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @fwd_strong_zero_different(%x: f64, %d1: f64, %d2: f64) -> (f64, f64) {
    %r1 = enzyme.fwddiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>], strong_zero = true}> : (f64, f64) -> f64
    %r2 = enzyme.fwddiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>], strong_zero = false}> : (f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @fwd_strong_zero_different
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @square
// CHECK-SAME: strong_zero = true
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @square
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @fwd_strong_zero_equal(%x: f64, %d1: f64, %d2: f64) -> (f64, f64) {
    %r1 = enzyme.fwddiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>], strong_zero = true}> : (f64, f64) -> f64
    %r2 = enzyme.fwddiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>], strong_zero = true}> : (f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @fwd_strong_zero_equal
// CHECK: enzyme.fwddiff @square
// CHECK-SAME: width = 2
// CHECK-SAME: strong_zero = true
// CHECK-NOT: enzyme.fwddiff
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @fwd_already_wide(%x: f64, %d1: tensor<2xf64>, %d2: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
    %r1 = enzyme.fwddiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>], width = 2 : i64}> : (f64, tensor<2xf64>) -> tensor<2xf64>
    %r2 = enzyme.fwddiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>], width = 2 : i64}> : (f64, tensor<2xf64>) -> tensor<2xf64>
    return %r1, %r2 : tensor<2xf64>, tensor<2xf64>
  }
}
// CHECK-LABEL: func.func @fwd_already_wide
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @square
// CHECK-SAME: width = 2
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @square
// CHECK-SAME: width = 2
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @rev_strong_zero_different(%x: f64, %d1: f64, %d2: f64) -> (f64, f64) {
    %r1 = enzyme.autodiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], strong_zero = true}> : (f64, f64) -> f64
    %r2 = enzyme.autodiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], strong_zero = false}> : (f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @rev_strong_zero_different
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @square
// CHECK-SAME: strong_zero = true
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @square
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @rev_strong_zero_equal(%x: f64, %d1: f64, %d2: f64) -> (f64, f64) {
    %r1 = enzyme.autodiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], strong_zero = true}> : (f64, f64) -> f64
    %r2 = enzyme.autodiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], strong_zero = true}> : (f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @rev_strong_zero_equal
// CHECK: enzyme.autodiff @square
// CHECK-SAME: width = 2
// CHECK-SAME: strong_zero = true
// CHECK-NOT: enzyme.autodiff
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @rev_atomic_add_different(%x: f64, %d1: f64, %d2: f64) -> (f64, f64) {
    %r1 = enzyme.autodiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], atomic_add = true}> : (f64, f64) -> f64
    %r2 = enzyme.autodiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], atomic_add = false}> : (f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @rev_atomic_add_different
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @square
// CHECK-SAME: atomic_add = true
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @square
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @rev_atomic_add_equal(%x: f64, %d1: f64, %d2: f64) -> (f64, f64) {
    %r1 = enzyme.autodiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], atomic_add = true}> : (f64, f64) -> f64
    %r2 = enzyme.autodiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], atomic_add = true}> : (f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @rev_atomic_add_equal
// CHECK: enzyme.autodiff @square
// CHECK-SAME: width = 2
// CHECK-SAME: atomic_add = true
// CHECK-NOT: enzyme.autodiff
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  func.func @rev_already_wide(%x: f64, %d1: tensor<2xf64>, %d2: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
    %r1 = enzyme.autodiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], width = 2 : i64}> : (f64, tensor<2xf64>) -> tensor<2xf64>
    %r2 = enzyme.autodiff @square(%x, %d2) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>], width = 2 : i64}> : (f64, tensor<2xf64>) -> tensor<2xf64>
    return %r1, %r2 : tensor<2xf64>, tensor<2xf64>
  }
}
// CHECK-LABEL: func.func @rev_already_wide
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @square
// CHECK-SAME: width = 2
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @square
// CHECK-SAME: width = 2
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

// Two active arguments, an intervening constant argument, and a retained primal.
module {
  func.func @muladd(%x: f64, %c: f64, %y: f64) -> f64 {
    %v = arith.mulf %x, %y : f64
    %r = arith.addf %v, %c : f64
    return %r : f64
  }
  func.func @reverse_result_mapping(%x: f64, %c: f64, %y: f64, %s1: f64, %s2: f64) -> (f64, f64, f64, f64, f64, f64) {
    %p1, %dx1, %dy1 = enzyme.autodiff @muladd(%x, %c, %y, %s1) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (f64, f64, f64, f64) -> (f64, f64, f64)
    %p2, %dx2, %dy2 = enzyme.autodiff @muladd(%x, %c, %y, %s2) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (f64, f64, f64, f64) -> (f64, f64, f64)
    return %p1, %dx1, %dy1, %p2, %dx2, %dy2 : f64, f64, f64, f64, f64, f64
  }
}
// CHECK-LABEL: func.func @reverse_result_mapping
// CHECK: %[[D:.*]]:3 = enzyme.autodiff @muladd
// CHECK-SAME: width = 2
// CHECK: %[[DX1:.*]] = enzyme.extract %[[D]]#1[0]
// CHECK: %[[DX2:.*]] = enzyme.extract %[[D]]#1[1]
// CHECK: %[[DY1:.*]] = enzyme.extract %[[D]]#2[0]
// CHECK: %[[DY2:.*]] = enzyme.extract %[[D]]#2[1]
// CHECK: return %[[D]]#0, %[[DX1]], %[[DY1]], %[[D]]#0, %[[DX2]], %[[DY2]]

// -----

module {
  func.func private @unknown(memref<1xf64>)
  func.func @load_scale(%p: memref<1xf64>, %x: f64) -> f64 {
    %i = arith.constant 0 : index
    %c = memref.load %p[%i] : memref<1xf64>
    %v = arith.mulf %x, %c : f64
    return %v : f64
  }
  func.func @write(%p: memref<1xf64>, %x: f64, %dx1: f64, %dx2: f64) -> (f64, f64) {
    %i = arith.constant 0 : index
    %r1 = enzyme.fwddiff @load_scale(%p, %x, %dx1) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    memref.store %x, %p[%i] : memref<1xf64>
    %r2 = enzyme.fwddiff @load_scale(%p, %x, %dx2) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @write
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func private @unknown(memref<1xf64>)
  func.func @load_scale(%p: memref<1xf64>, %x: f64) -> f64 {
    %i = arith.constant 0 : index
    %c = memref.load %p[%i] : memref<1xf64>
    %v = arith.mulf %x, %c : f64
    return %v : f64
  }
  func.func @alias_write(%p: memref<1xf64>, %x: f64, %dx1: f64, %dx2: f64) -> (f64, f64) {
    %i = arith.constant 0 : index
    %r1 = enzyme.fwddiff @load_scale(%p, %x, %dx1) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    %alias = memref.cast %p : memref<1xf64> to memref<?xf64>
    memref.store %x, %alias[%i] : memref<?xf64>
    %r2 = enzyme.fwddiff @load_scale(%p, %x, %dx2) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @alias_write
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func private @unknown(memref<1xf64>)
  func.func @load_scale(%p: memref<1xf64>, %x: f64) -> f64 {
    %i = arith.constant 0 : index
    %c = memref.load %p[%i] : memref<1xf64>
    %v = arith.mulf %x, %c : f64
    return %v : f64
  }
  func.func @free(%p: memref<1xf64>, %x: f64, %dx1: f64, %dx2: f64) -> (f64, f64) {
    %i = arith.constant 0 : index
    %r1 = enzyme.fwddiff @load_scale(%p, %x, %dx1) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    memref.dealloc %p : memref<1xf64>
    %r2 = enzyme.fwddiff @load_scale(%p, %x, %dx2) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @free
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func private @unknown(memref<1xf64>)
  func.func @load_scale(%p: memref<1xf64>, %x: f64) -> f64 {
    %i = arith.constant 0 : index
    %c = memref.load %p[%i] : memref<1xf64>
    %v = arith.mulf %x, %c : f64
    return %v : f64
  }
  func.func @unknown_call(%p: memref<1xf64>, %x: f64, %dx1: f64, %dx2: f64) -> (f64, f64) {
    %i = arith.constant 0 : index
    %r1 = enzyme.fwddiff @load_scale(%p, %x, %dx1) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    func.call @unknown(%p) : (memref<1xf64>) -> ()
    %r2 = enzyme.fwddiff @load_scale(%p, %x, %dx2) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @unknown_call
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func private @unknown(memref<1xf64>)
  func.func @load_scale(%p: memref<1xf64>, %x: f64) -> f64 {
    %i = arith.constant 0 : index
    %c = memref.load %p[%i] : memref<1xf64>
    %v = arith.mulf %x, %c : f64
    return %v : f64
  }
  func.func @read_only(%p: memref<1xf64>, %x: f64, %dx1: f64, %dx2: f64) -> (f64, f64) {
    %i = arith.constant 0 : index
    %r1 = enzyme.fwddiff @load_scale(%p, %x, %dx1) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    %ignored = memref.load %p[%i] : memref<1xf64>
    %r2 = enzyme.fwddiff @load_scale(%p, %x, %dx2) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    return %r1, %r2 : f64, f64
  }
}
// CHECK-LABEL: func.func @read_only
// CHECK: enzyme.fwddiff @load_scale
// CHECK-SAME: width = 2
// CHECK-NOT: enzyme.fwddiff
// CHECK: return

// -----

module {
  func.func @product(%x: f64, %y: f64) -> f64 {
    %r = arith.mulf %x, %y : f64
    return %r : f64
  }
  func.func @fwd_different_input_activity(%x: f64, %y: f64, %seed: f64) -> (f64, f64) {
    %d1 = enzyme.fwddiff @product(%x, %seed, %y) <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64, f64) -> f64
    %d2 = enzyme.fwddiff @product(%x, %y, %seed) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64, f64) -> f64
    return %d1, %d2 : f64, f64
  }
}
// CHECK-LABEL: func.func @fwd_different_input_activity
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @product
// CHECK-SAME: activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>]
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @product
// CHECK-SAME: activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>]
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func @product(%x: f64, %y: f64) -> f64 {
    %r = arith.mulf %x, %y : f64
    return %r : f64
  }
  func.func @rev_different_input_activity(%x: f64, %y: f64, %seed: f64) -> (f64, f64) {
    %d1 = enzyme.autodiff @product(%x, %y, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64, f64) -> f64
    %d2 = enzyme.autodiff @product(%x, %y, %seed) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64, f64) -> f64
    return %d1, %d2 : f64, f64
  }
}
// CHECK-LABEL: func.func @rev_different_input_activity
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @product
// CHECK-SAME: activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.autodiff @product
// CHECK-SAME: activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>]
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %v = arith.mulf %x, %x : f64
    return %v : f64
  }
  // The second seed is produced by the first request, so it cannot move up.
  func.func @dependent_seed(%x: f64, %dx: f64) -> (f64, f64) {
    %d1 = enzyme.fwddiff @square(%x, %dx) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    %d2 = enzyme.fwddiff @square(%x, %d1) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    return %d1, %d2 : f64, f64
  }
  // A seed defined above the containing region dominates both requests.
  func.func @outer_seed(%x: f64, %cond: i1) -> (f64, f64) {
    %dx = arith.addf %x, %x : f64
    %d:2 = scf.if %cond -> (f64, f64) {
      %d1 = enzyme.fwddiff @square(%x, %dx) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
      %d2 = enzyme.fwddiff @square(%x, %dx) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
      scf.yield %d1, %d2 : f64, f64
    } else {
      scf.yield %x, %x : f64, f64
    }
    return %d#0, %d#1 : f64, f64
  }
}
// CHECK-LABEL: func.func @dependent_seed
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff
// CHECK-NOT: enzyme.concat
// CHECK: return
// CHECK-LABEL: func.func @outer_seed
// CHECK: enzyme.fwddiff
// CHECK-SAME: width = 2
// CHECK-NOT: enzyme.fwddiff
// CHECK: return

// -----

module {
  func.func private @unknown(f64) -> f64
  func.func @external_callee(%x: f64, %s1: f64, %s2: f64) -> (f64, f64) {
    %d1 = enzyme.fwddiff @unknown(%x, %s1) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    %d2 = enzyme.fwddiff @unknown(%x, %s2) <{activity = [#enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    return %d1, %d2 : f64, f64
  }
  // An intervening differentiation of an external function has unknown effects.
  func.func @load_scale(%p: memref<1xf64>, %x: f64) -> f64 {
    %i = arith.constant 0 : index
    %c = memref.load %p[%i] : memref<1xf64>
    %r = arith.mulf %c, %x : f64
    return %r : f64
  }
  func.func @external_diff_between(%p: memref<1xf64>, %x: f64, %s1: f64, %s2: f64) -> (f64, f64, f64) {
    %d1 = enzyme.fwddiff @load_scale(%p, %x, %s1) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    %d2 = enzyme.autodiff @unknown(%x, %s1) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
    %d3 = enzyme.fwddiff @load_scale(%p, %x, %s2) <{activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity = [#enzyme.activity<enzyme_dupnoneed>]}> : (memref<1xf64>, f64, f64) -> f64
    return %d1, %d2, %d3 : f64, f64, f64
  }
}
// CHECK-LABEL: func.func @external_callee
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @unknown
// CHECK: enzyme.fwddiff @unknown
// CHECK-NOT: enzyme.concat
// CHECK: return
// CHECK-LABEL: func.func @external_diff_between
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK: enzyme.autodiff @unknown
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  func.func @square(%x: f64) -> f64 {
    %r = arith.mulf %x, %x : f64
    return %r : f64
  }
  func.func @dependent_primals(%x: f64, %s1: f64, %s2: f64) -> (f64, f64, f64, f64) {
    %p1, %d1 = enzyme.fwddiff @square(%x, %s1) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dup>]}> : (f64, f64) -> (f64, f64)
    %p2, %d2 = enzyme.fwddiff @square(%x, %s2) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dup>]}> : (f64, f64) -> (f64, f64)
    %q1 = enzyme.fwddiff @square(%p1, %s1) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    %q2 = enzyme.fwddiff @square(%p1, %s2) <{activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (f64, f64) -> f64
    return %d1, %d2, %q1, %q2 : f64, f64, f64, f64
  }
}
// CHECK-LABEL: func.func @dependent_primals
// CHECK: %[[FIRST:.*]]:2 = enzyme.fwddiff @square
// CHECK-SAME: width = 2
// CHECK: enzyme.fwddiff @square(%[[FIRST]]#0,
// CHECK-SAME: width = 2
// CHECK: return

// -----

module {
  llvm.func @clobber(!llvm.ptr)
  func.func @load_scale(%p: !llvm.ptr, %x: f64) -> f64 {
    %c = llvm.load %p : !llvm.ptr -> f64
    %r = arith.mulf %c, %x : f64
    return %r : f64
  }
  func.func @llvm_call_between(%p: !llvm.ptr, %x: f64, %s1: f64, %s2: f64) -> (f64, f64) {
    %d1 = enzyme.fwddiff @load_scale(%p, %x, %s1) <{activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (!llvm.ptr, f64, f64) -> f64
    llvm.call @clobber(%p) : (!llvm.ptr) -> ()
    %d2 = enzyme.fwddiff @load_scale(%p, %x, %s2) <{activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (!llvm.ptr, f64, f64) -> f64
    return %d1, %d2 : f64, f64
  }
}
// CHECK-LABEL: func.func @llvm_call_between
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK: llvm.call @clobber
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: return

// -----

module {
  llvm.func @clobber(!llvm.ptr)
  func.func @load_scale(%p: !llvm.ptr, %x: f64) -> f64 {
    llvm.call @clobber(%p) : (!llvm.ptr) -> ()
    %c = llvm.load %p : !llvm.ptr -> f64
    %r = arith.mulf %c, %x : f64
    return %r : f64
  }
  func.func @llvm_call_in_callee(%p: !llvm.ptr, %x: f64, %s1: f64, %s2: f64) -> (f64, f64) {
    %d1 = enzyme.fwddiff @load_scale(%p, %x, %s1) <{activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (!llvm.ptr, f64, f64) -> f64
    %d2 = enzyme.fwddiff @load_scale(%p, %x, %s2) <{activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>]}> : (!llvm.ptr, f64, f64) -> f64
    return %d1, %d2 : f64, f64
  }
}
// CHECK-LABEL: func.func @llvm_call_in_callee
// CHECK-NOT: enzyme.concat
// CHECK: enzyme.fwddiff @load_scale
// CHECK: enzyme.fwddiff @load_scale
// CHECK-NOT: enzyme.concat
// CHECK: return
