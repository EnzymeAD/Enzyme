// RUN: %eopt --enzyme %s | FileCheck %s

// A block argument's tangent goes right after it. When the active argument is
// not the last one and the block already has predecessors -- the header of
// any loop that carries an active value ahead of a counter, as mem2reg leaves
// it -- the tangent was inserted with the iterator overload of
// Block::insertArgument, which asserts:
//
//   cannot insert arguments to blocks with predecessors
//
// The branches into the block are rebuilt afterwards with each tangent after
// its primal, so inserting by index is fine.

module {
  func.func private @f(%x: f64, %n: i32) -> f64 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %z = arith.constant 0.0 : f64
    cf.br ^loop(%z, %c0 : f64, i32)
  ^loop(%s: f64, %i: i32):
    %go = arith.cmpi slt, %i, %n : i32
    cf.cond_br %go, ^body, ^exit
  ^body:
    %s2 = arith.addf %s, %x : f64
    %i2 = arith.addi %i, %c1 : i32
    cf.br ^loop(%s2, %i2 : f64, i32)
  ^exit:
    return %s : f64
  }

  func.func @df(%x: f64, %dx: f64, %n: i32) -> f64 {
    %r = enzyme.fwddiff @f(%x, %dx, %n) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64, i32) -> f64
    return %r : f64
  }
}

// The tangent of %s sits between %s and %i; the entry edge passes a zero
// tangent for the constant start and the back edge passes the tangent of %s2.
// CHECK: func.func private @fwddiffef(%[[x:.+]]: f64, %[[dx:.+]]: f64, %[[n:.+]]: i32) -> f64
// CHECK:         %[[z:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:         %[[dz:.+]] = arith.constant 0.000000e+00 : f64
// CHECK:         cf.br ^[[loop:.+]](%[[z]], %[[dz]], %{{.+}} : f64, f64, i32)
// CHECK:       ^[[loop]](%[[s:.+]]: f64, %[[ds:.+]]: f64, %[[i:.+]]: i32):
// CHECK:         %[[ds2:.+]] = arith.addf %[[ds]], %[[dx]]
// CHECK:         %[[s2:.+]] = arith.addf %[[s]], %[[x]]
// CHECK:         %[[i2:.+]] = arith.addi %[[i]],
// CHECK:         cf.br ^[[loop]](%[[s2]], %[[ds2]], %[[i2]] : f64, f64, i32)
// CHECK:         return %[[ds]] : f64
