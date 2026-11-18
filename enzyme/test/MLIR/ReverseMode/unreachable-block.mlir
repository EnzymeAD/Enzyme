// RUN: %eopt --enzyme %s | FileCheck %s

// A block nothing branches to was taken for the entry block: its reverse got
// the function's return, built from the function's argument activities and
// the block's own arguments, and failed the verifier with
//
//   'func.return' op has 0 operands, but enclosing function returns 1
//
// Its reverse never runs; it now branches to the reverse of the entry block.

module {
  func.func private @f(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    cf.br ^exit(%y : f64)
  ^dead:
    %z = arith.addf %x, %x : f64
    cf.br ^exit(%z : f64)
  ^exit(%r: f64):
    return %r : f64
  }
  func.func @df(%x: f64, %seed: f64) -> f64 {
    %d = enzyme.autodiff @f(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (f64, f64) -> f64
    return %d : f64
  }
}

// CHECK-LABEL: func.func private @diffef(
// CHECK:       ^bb1:  // no predecessors
// CHECK:         cf.br ^bb2(
// CHECK:       ^bb4:  // no predecessors
// CHECK:         cf.br ^bb5
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb3, ^bb4
// CHECK:         return %{{.+}} : f64
