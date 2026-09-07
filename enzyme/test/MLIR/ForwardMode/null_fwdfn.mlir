// RUN: %eopt --enzyme --verify-diagnostics %s

// When CreateForwardDiff fails for a callee of a func.call inside a
// forward-differentiated function, emit a diagnostic instead of
// dereferencing the null FunctionOpInterface and segfaulting.

module {
  func.func private @ext(f64) -> f64
  func.func @inner(%x: f64) -> f64 {
    // expected-error @below {{cannot differentiate a call to a function without a body and without a registered derivative: "ext"}}
    %r = func.call @ext(%x) : (f64) -> f64
    return %r : f64
  }
  func.func @outer(%x: f64) -> f64 {
    // expected-error @below {{failed to create forward-mode derivative for callee "inner"}}
    %r = func.call @inner(%x) : (f64) -> f64
    return %r : f64
  }
  func.func @d_outer(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @outer(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}
