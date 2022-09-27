// RUN: %eopt --enzyme --verify-diagnostics %s

module {
  func.func @unsupported(%x : f64) -> f64 {
    // We are not going to differentiate nested funcs, at least for a while.
    // expected-error @below {{could not compute the adjoint for this operation}}
    func.func private @foo()
    %c = arith.constant 42.0 : f64
    return %c : f64
  }
  func.func @diff(%x : f64, %dx : f64) -> f64 {
    %r = enzyme.fwddiff @unsupported(%x, %dx) { activity=[#enzyme<activity enzyme_dup>] } : (f64, f64) -> (f64)
    return %r : f64
  }
}
