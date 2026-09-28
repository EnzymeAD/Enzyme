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

// The mutually recursive @f and @g are differentiated in split mode until a
// call reaches a rule that is still being derived: the rule caches its values
// as typed values, which cannot contain themselves, so that call takes the
// combined-mode path (@diffeg / @diffef_0, which recompute the primal in the
// reverse and recurse into each other).

// CHECK:  func.func private @f_rev(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0:2 = call @g_reverse_rule_primal(%arg0) : (f32) -> (f32, f32)
// CHECK-NEXT:    %1 = call @g_reverse_rule_reverse(%arg1, %0#1) : (f32, f32) -> f32
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffeg(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = call @f(%arg0) : (f32) -> f32
// CHECK-NEXT:    %1 = call @diffef_0(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @diffef_0(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = call @g(%arg0) : (f32) -> f32
// CHECK-NEXT:    %1 = call @diffeg(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT:    return %1 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @f_reverse_rule_primal(%arg0: f32) -> (f32, f32) {
// CHECK-NEXT:    %0 = call @g(%arg0) : (f32) -> f32
// CHECK-NEXT:    return %0, %arg0 : f32, f32
// CHECK-NEXT:  }

// CHECK:  func.func private @f_reverse_rule_reverse(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = call @diffeg(%arg1, %arg0) : (f32, f32) -> f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }

// CHECK:  func.func private @g_reverse_rule_primal(%arg0: f32) -> (f32, f32) {
// CHECK-NEXT:    %0:2 = call @f_reverse_rule_primal(%arg0) : (f32) -> (f32, f32)
// CHECK-NEXT:    return %0#0, %0#1 : f32, f32
// CHECK-NEXT:  }

// CHECK:  func.func private @g_reverse_rule_reverse(%arg0: f32, %arg1: f32) -> f32 {
// CHECK-NEXT:    %0 = call @f_reverse_rule_reverse(%arg0, %arg1) : (f32, f32) -> f32
// CHECK-NEXT:    return %0 : f32
// CHECK-NEXT:  }
