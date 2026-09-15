// RUN: %eopt --enzyme --verify-diagnostics %s

// Differentiating a function that has no body at all fails the same way a
// call to one inside a body does.

module {
  // expected-error @below {{cannot differentiate a function without a body: "ext"}}
  func.func private @ext(f64) -> f64
  func.func @dext(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @ext(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}
