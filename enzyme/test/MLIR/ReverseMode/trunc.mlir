// RUN: %eopt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse %s | FileCheck %s --check-prefix=FIN

module {
  func.func @f(%x: f64) -> f32 {
    %next = arith.truncf %x : f64 to f32
    return %next : f32
  }

  func.func @dsquare(%x: f64, %dr: f32) -> f64 {
    %r = enzyme.autodiff @f(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f32) -> f64
    return %r : f64
  }
}

// FIN:  func.func private @diffef(%[[x:.+]]: f64, %[[dx:.+]]: f32) -> f64 {
// FIN-NEXT:    %[[res:.+]] = arith.extf %[[dx]] : f32 to f64
// FIN-NEXT:    return %[[res]] : f64
// FIN-NEXT:  }
