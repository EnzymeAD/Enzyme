// RUN: %eopt %s --enzyme-wrap="infn=main retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s
// RUN: %eopt %s --enzyme-wrap="infn=main retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined lower-custom-rules=false" | FileCheck %s --check-prefix=KEEP

// A call is differentiated in split mode, through a rule derived for the
// callee. Like the enzyme pass, enzyme-wrap lowers the rule and the calls to
// it, so that the passes after it see functions and calls only.

module {
  func.func private @sq(%x: f64) -> f64 {
    %y = arith.mulf %x, %x : f64
    return %y : f64
  }

  func.func @main(%x: f64) -> f64 {
    %y = func.call @sq(%x) : (f64) -> f64
    return %y : f64
  }
}

// CHECK-NOT: enzyme.
// CHECK: func.func @main(%[[x:.+]]: f64, %[[dy:.+]]: f64) -> f64
// CHECK: call @sq_reverse_rule_primal(%[[x]])
// CHECK: call @sq_reverse_rule_reverse(%[[dy]],
// CHECK-NOT: enzyme.

// KEEP: enzyme.custom_reverse_rule @sq_reverse_rule
// KEEP: enzyme.call_augmented_primal @sq_reverse_rule
// KEEP: enzyme.call_custom_reverse @sq_reverse_rule
