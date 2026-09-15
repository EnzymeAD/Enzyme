// RUN: %eopt --enzyme --verify-diagnostics %s

// Differentiating a function that has no body at all fails the same way a
// call to one inside a body does.

module {
  // expected-error @below {{cannot differentiate a function without a body: "ext"}}
  func.func private @ext(f64) -> f64
  func.func @dext(%x: f64, %dr: f64) -> f64 {
    %g = enzyme.autodiff @ext(%x, %dr) { activity=[#enzyme<activity enzyme_active>], ret_activity=[#enzyme<activity enzyme_activenoneed>] } : (f64, f64) -> f64
    return %g : f64
  }
}
