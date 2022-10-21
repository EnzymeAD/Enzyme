// RUN: %eopt --enzyme --verify-diagnostics --allow-unregistered-dialect %s

module {
  func.func @unsupported(%x : f64) -> f64 {
    // We cannot differentiate active (from operands+results) ops from unknown dialects.
    // expected-error @below {{could not compute the adjoint for this operation}}
    %r = "test.foo"(%x) : (f64) -> f64
    return %r : f64
  }
  func.func @diff(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @unsupported(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}
