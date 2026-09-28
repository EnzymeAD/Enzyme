// RUN: %eopt --enzyme-wrap="infn=f outfn=f_rev retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --lower-enzyme-custom-rules-to-func --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

module {
  func.func @f(%arg0: f32) -> f32 {
    %0 = call @g(%arg0) : (f32) -> f32
    return %0 : f32
  }

  func.func @g(%arg0: f32) -> f32 {
    %0 = call @f(%arg0) : (f32) -> f32
    return %0 : f32
  }
}

// The custom rules for the mutually recursive @f and @g are lowered to
// func.func pairs. Because the recursion is unbounded, the tape cannot be
// flattened and is threaded through as an opaque !enzyme.Tape.

// CHECK:  func.func private @f_rev(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0:2 = call @g_reverse_rule_primal(%arg0) : (f32) -> (f32, !enzyme.Tape)
// CHECK-NEXT:    %1 = call @g_reverse_rule_reverse(%arg1, %0#1) : (f32, !enzyme.Tape) -> f32
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @f_reverse_rule_primal(%arg0: f32) -> (f32, !enzyme.Tape) {
// CHECK-NEXT:    %0:2 = call @g_reverse_rule_primal(%arg0) : (f32) -> (f32, !enzyme.Tape)
// CHECK-NEXT:    return %0#0, %0#1 : f32, !enzyme.Tape
// CHECK-NEXT:  }

// CHECK:  func.func private @f_reverse_rule_reverse(%arg0: f32, %arg1: !enzyme.Tape) -> f32 {
// CHECK-NEXT:    %0 = call @g_reverse_rule_reverse(%arg0, %arg1) : (f32, !enzyme.Tape) -> f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @g_reverse_rule_primal(%arg0: f32) -> (f32, !enzyme.Tape) {
// CHECK-NEXT:    %0:2 = call @f_reverse_rule_primal(%arg0) : (f32) -> (f32, !enzyme.Tape)
// CHECK-NEXT:    return %0#0, %0#1 : f32, !enzyme.Tape
// CHECK-NEXT:  }

// CHECK:  func.func private @g_reverse_rule_reverse(%arg0: f32, %arg1: !enzyme.Tape) -> f32 {
// CHECK-NEXT:    %0 = call @f_reverse_rule_reverse(%arg0, %arg1) : (f32, !enzyme.Tape) -> f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }
