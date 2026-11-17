// RUN: %eopt --enzyme %s | FileCheck %s

// In reverse mode only a pointer block argument gets a shadow argument (the
// adjoints of the others are kept apart from the primal blocks). Branches had
// no reverse-mode handling at all, so nothing passed that shadow along the
// edges into the block and the result failed the verifier with
//
//   branch has 3 operands for successor #0, but target block has 4
//
// The forward pass now rebuilds a branch into such a block with each pointer's
// shadow right after it.

module {
  func.func private @walk(%p: !llvm.ptr, %n: i32) -> f64 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %z = arith.constant 0.0 : f64
    cf.br ^loop(%z, %p, %c0 : f64, !llvm.ptr, i32)
  ^loop(%s: f64, %q: !llvm.ptr, %i: i32):
    %go = arith.cmpi slt, %i, %n : i32
    cf.cond_br %go, ^body, ^exit
  ^body:
    %v = llvm.load %q : !llvm.ptr -> f64
    %s2 = arith.addf %s, %v : f64
    %q2 = llvm.getelementptr %q[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %i2 = arith.addi %i, %c1 : i32
    cf.br ^loop(%s2, %q2, %i2 : f64, !llvm.ptr, i32)
  ^exit:
    return %s : f64
  }
  func.func @dwalk(%p: !llvm.ptr, %dp: !llvm.ptr, %n: i32, %seed: f64) {
    enzyme.autodiff @walk(%p, %dp, %n, %seed) <{ activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!llvm.ptr, !llvm.ptr, i32, f64) -> ()
    return
  }
}

// The f64 sum has no shadow argument; the pointer's shadow follows it. The
// entry edge passes %dp and the back edge the shadow gep.
// CHECK: func.func private @diffewalk(%[[p:.+]]: !llvm.ptr, %[[dp:.+]]: !llvm.ptr, %{{.+}}: i32, %{{.+}}: f64)
// CHECK:         cf.br ^[[loop:.+]](%{{.+}}, %[[p]], %[[dp]], %{{.+}} : f64, !llvm.ptr, !llvm.ptr, i32)
// CHECK:       ^[[loop]](%{{.+}}: f64, %[[q:.+]]: !llvm.ptr, %[[dq:.+]]: !llvm.ptr, %{{.+}}: i32):
// CHECK:         %[[dq2:.+]] = llvm.getelementptr %[[dq]][1]
// CHECK:         %[[q2:.+]] = llvm.getelementptr %[[q]][1]
// CHECK:         cf.br ^[[loop]](%{{.+}}, %[[q2]], %[[dq2]], %{{.+}} : f64, !llvm.ptr, !llvm.ptr, i32)
