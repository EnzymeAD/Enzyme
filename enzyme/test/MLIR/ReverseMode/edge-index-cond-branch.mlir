// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// The backward pass of a block entered along several edges pops which one was
// taken. The index used to be pushed in front of each predecessor's
// terminator, so a conditional branch pushed it even when it went elsewhere,
// and a later pop could read one of those: ^merge below is entered from ^head
// and ^body, both conditional, once per outer iteration, and its gradient
// came out wrong. An edge from a conditional branch now passes its index to
// the block, which pushes it.

module {
  func.func private @f(%x: f64, %n: i32) -> f64 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %one = arith.constant 1.0 : f64
    cf.br ^outer(%one, %c0 : f64, i32)
  ^outer(%s: f64, %j: i32):
    %go = arith.cmpi slt, %j, %c3 : i32
    cf.cond_br %go, ^head(%s, %c0 : f64, i32), ^done(%s : f64)
  ^head(%t: f64, %i: i32):
    %more = arith.cmpi slt, %i, %n : i32
    cf.cond_br %more, ^body, ^merge(%t : f64)
  ^body:
    %hit = arith.cmpi eq, %i, %j : i32
    %t2 = arith.mulf %t, %x : f64
    %i2 = arith.addi %i, %c1 : i32
    cf.cond_br %hit, ^merge(%t2 : f64), ^head(%t2, %i2 : f64, i32)
  ^merge(%u: f64):
    %u2 = arith.mulf %u, %x : f64
    %j2 = arith.addi %j, %c1 : i32
    cf.br ^outer(%u2, %j2 : f64, i32)
  ^done(%r: f64):
    return %r : f64
  }
  func.func @df(%x: f64, %n: i32, %seed: f64) -> f64 {
    %d = enzyme.autodiff @f(%x, %n, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, i32, f64) -> f64
    return %d : f64
  }
}

// ^outer is only entered by unconditional branches: the index is still pushed
// in front of them.
// CHECK-LABEL: func.func private @diffef(
// CHECK:         "enzyme.push"(%[[OUTER:.+]], %c0_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.br ^bb1(
// CHECK:         cf.cond_br %{{.+}}, ^bb2(%{{.+}}, %c0_i32, %c0_i32 : f64, i32, i32), ^bb5
// CHECK-NEXT:  ^bb2(%{{.+}}: f64, %{{.+}}: i32, %[[HEAD:.+]]: i32):
// CHECK-NEXT:    "enzyme.push"(%{{.+}}, %[[HEAD]]) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK:         cf.cond_br %{{.+}}, ^bb3, ^bb4(%{{.+}}, %c0_i32 : f64, i32)
// CHECK:         cf.cond_br %{{.+}}, ^bb4(%{{.+}}, %c-1_i32 : f64, i32), ^bb2(%{{.+}}, %{{.+}}, %c-1_i32 : f64, i32, i32)
// CHECK-NEXT:  ^bb4(%{{.+}}: f64, %[[MERGE:.+]]: i32):
// CHECK-NEXT:    "enzyme.push"(%{{.+}}, %[[MERGE]]) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK:         "enzyme.push"(%[[OUTER]], %c-1_i32) : (!enzyme.Cache<i32>, i32) -> ()
// CHECK-NEXT:    cf.br ^bb1(
