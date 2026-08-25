// RUN: %eopt --enzyme --verify-diagnostics %s

// When CreateReverseDiff fails for a callee of a func.call inside a
// reverse-differentiated function, emit a diagnostic instead of
// dereferencing the null FunctionOpInterface and segfaulting.

module {
  func.func private @ext(f64) -> f64
  func.func @inner(%x: f64) -> f64 {
    // expected-error @below {{cannot differentiate a call to a function without a body and without a registered derivative: "ext"}}
    %r = func.call @ext(%x) : (f64) -> f64
    return %r : f64
  }
  func.func @outer(%x: f64) -> f64 {
    // expected-error @below {{failed to create reverse-mode adjoint for callee "inner"}}
    %r = func.call @inner(%x) : (f64) -> f64
    return %r : f64
  }
  func.func @d_outer(%x: f64, %dr: f64) -> f64 {
    %g = enzyme.autodiff @outer(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %g : f64
  }
}
