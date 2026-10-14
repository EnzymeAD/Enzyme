// RUN: %eopt %s --split-input-file --enzyme --canonicalize --lower-enzyme-custom-rules-to-func --remove-unnecessary-enzyme-ops --enzyme-simplify-math --verify-diagnostics | FileCheck %s
// RUN: %eopt %s --split-input-file --mlir-print-op-generic | FileCheck %s --check-prefix=PARSE

module {
  func.func @mul(%a: f32, %b: f32) -> f32 {
    %0 = arith.mulf %a, %b : f32
    %1 = math.exp %0 : f32
    %2 = arith.addf %b, %1 : f32
    return %2 : f32
  }

  // Split mode
  func.func @main(%a: f32, %b: f32) -> (f32, f32, f32) {
    %r, %tape = enzyme.autodiff_split_mode.primal @mul(%a, %b) {
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    } : (f32, f32) -> (f32, !enzyme.Tape)

    // ---

    %dres = arith.constant 1.0 : f32
    %da, %db = enzyme.autodiff_split_mode.reverse @mul(%dres, %tape) {
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    } : (f32, !enzyme.Tape) -> (f32, f32)

    return %r, %da, %db : f32, f32, f32
  }
}

// CHECK:  func.func @main(%arg0: f32, %arg1: f32) -> (f32, f32, f32) {
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %0:3 = call @mul_reverse_rule_primal(%arg0, %arg1) : (f32, f32) -> (f32, f32, f32)
// CHECK-NEXT:    %1:2 = call @mul_reverse_rule_reverse(%cst, %0#1, %0#2) : (f32, f32, f32) -> (f32, f32)
// CHECK-NEXT:    return %0#0, %1#0, %1#1 : f32, f32, f32
// CHECK-NEXT:  }

// CHECK:  func.func private @mul_reverse_rule_primal(%arg0: f32, %arg1: f32) -> (f32, f32, f32) {
// CHECK-NEXT:    %0 = arith.mulf %arg0, %arg1 : f32
// CHECK-NEXT:    %1 = math.exp %0 : f32
// CHECK-NEXT:    %2 = arith.addf %arg1, %1 : f32
// CHECK-NEXT:    return %2, %arg0, %arg1 : f32, f32, f32
// CHECK-NEXT:  }

// The product feeding the exp is recomputed in the reverse rather than put on
// the tape.
// CHECK:  func.func private @mul_reverse_rule_reverse(%arg0: f32, %arg1: f32, %arg2: f32) -> (f32, f32) {
// CHECK-NEXT:    %0 = arith.mulf %arg1, %arg2 : f32
// CHECK-NEXT:    %1 = math.exp %0 fastmath<fast> : f32
// CHECK-NEXT:    %2 = arith.mulf %arg0, %1 fastmath<fast> : f32
// CHECK-NEXT:    %3 = arith.mulf %2, %arg2 fastmath<fast> : f32
// CHECK-NEXT:    %4 = arith.mulf %2, %arg1 fastmath<fast> : f32
// CHECK-NEXT:    %5 = arith.addf %arg0, %4 fastmath<fast> : f32
// CHECK-NEXT:    return %3, %5 : f32, f32
// CHECK-NEXT:  }

// -----

// Shared operations must be cloned in definition order into both regions.
module {
  enzyme.custom_reverse_rule @shared_rule {
    %two = arith.constant 2.0 : f64
    %four = arith.addf %two, %two : f64
    enzyme.custom_reverse_rule.augmented_primal (%x: f64) -> f64 {
      %y = arith.mulf %four, %x : f64
      enzyme.yield %y : f64
    }
    enzyme.custom_reverse_rule.reverse (%dy: f64) -> f64 {
      %dx = arith.mulf %four, %dy : f64
      enzyme.yield %dx : f64
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f64) -> f64
  }
}

// PARSE: "enzyme.custom_reverse_rule.augmented_primal"() <{function_type = (f64) -> f64}>
// PARSE: "enzyme.custom_reverse_rule.reverse"() <{function_type = (f64) -> f64}>

// CHECK-LABEL: func.func private @shared_rule_primal(%arg0: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg0, %cst : f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }
// CHECK-LABEL: func.func private @shared_rule_reverse(%arg0: f64) -> f64 {
// CHECK-NEXT:    %cst = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.mulf %arg0, %cst : f64
// CHECK-NEXT:    return %0 : f64
// CHECK-NEXT:  }

// -----

// The noneed variants use the same rule ABI as const and dup.
module {
  enzyme.custom_reverse_rule @constnoneed_rule {
    enzyme.custom_reverse_rule.augmented_primal (%x: f64) -> f64 {
      enzyme.yield %x : f64
    }
    enzyme.custom_reverse_rule.reverse () {
      enzyme.yield
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_constnoneed>],
    ret_activity = [#enzyme.activity<enzyme_constnoneed>],
    function_type = (f64) -> f64
  }

  enzyme.custom_reverse_rule @dupnoneed_rule {
    enzyme.custom_reverse_rule.augmented_primal (%x: !llvm.ptr, %dx: !llvm.ptr) {
      enzyme.yield
    }
    enzyme.custom_reverse_rule.reverse () {
      enzyme.yield
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_dupnoneed>],
    ret_activity = [],
    function_type = (!llvm.ptr) -> ()
  }
}

// PARSE: "enzyme.custom_reverse_rule.augmented_primal"() <{function_type = (f64) -> f64}>
// PARSE: "enzyme.custom_reverse_rule.reverse"() <{function_type = () -> ()}>
// PARSE: "enzyme.custom_reverse_rule.augmented_primal"() <{function_type = (!llvm.ptr, !llvm.ptr) -> ()}>
// PARSE: "enzyme.custom_reverse_rule.reverse"() <{function_type = () -> ()}>

// CHECK-LABEL: func.func private @constnoneed_rule_primal(%arg0: f64) -> f64 {
// CHECK-NEXT:    return %arg0 : f64
// CHECK-NEXT:  }
// CHECK-LABEL: func.func private @constnoneed_rule_reverse() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-LABEL: func.func private @dupnoneed_rule_primal(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-LABEL: func.func private @dupnoneed_rule_reverse() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

module {
  // expected-error @below {{custom reverse rules only support width 1}}
  func.func @square(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }
  func.func @main(%x: f64) -> f64 {
    // expected-error @below {{failed to create reverse-mode adjoint for callee "square"}}
    %y, %tape = enzyme.autodiff_split_mode.primal @square(%x) {
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>],
      width = 2
    } : (f64) -> (f64, !enzyme.Tape)
    return %y : f64
  }
}
