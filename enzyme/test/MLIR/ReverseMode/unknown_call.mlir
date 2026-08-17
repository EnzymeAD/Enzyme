// RUN: %eopt --enzyme --verify-diagnostics %s

// The reverse pass refuses a call to a bodyless function without a
// registered derivative the same way the forward pass does.

module {
  func.func private @ext(f64) -> f64
  func.func @sq(%x: f64) -> f64 {
    // expected-error @below {{cannot differentiate a call to a function without a body and without a registered derivative: "ext"}}
    %r = func.call @ext(%x) : (f64) -> f64
    %m = arith.mulf %r, %r : f64
    return %m : f64
  }
  func.func @dsq(%x: f64, %dr: f64) -> f64 {
    %g = enzyme.autodiff @sq(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %g : f64
  }
}
