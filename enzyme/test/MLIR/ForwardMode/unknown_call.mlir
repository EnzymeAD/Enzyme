// RUN: %eopt --enzyme --verify-diagnostics %s

// A call to a function with no body and no registered derivative has no
// tangent to construct; refusing loudly beats a silent zero or a crash.

module {
  func.func private @ext(f64) -> f64
  func.func @sq(%x: f64) -> f64 {
    // expected-error @below {{cannot differentiate a call to a function without a body and without a registered derivative: "ext"}}
    %r = func.call @ext(%x) : (f64) -> f64
    %m = arith.mulf %r, %r : f64
    return %m : f64
  }
  func.func @dsq(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @sq(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (f64, f64) -> f64
    return %r : f64
  }
}
