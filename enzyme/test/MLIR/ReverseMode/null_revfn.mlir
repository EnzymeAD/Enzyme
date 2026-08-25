// RUN: %eopt --enzyme --verify-diagnostics %s

// When CreateReverseDiff fails for a callee of a func.call inside a
// reverse-differentiated function, emit a diagnostic instead of
// dereferencing the null FunctionOpInterface and segfaulting.
//
// arith.remf has no registered reverse-mode derivative, so
// CreateReverseDiff(@inner) fails, returns null, and the new guard in
// callReverseHandler emits the "failed to create" error rather than
// crashing with a SIGSEGV.

module {
  func.func @inner(%x: f64) -> f64 {
    // expected-error @below {{could not compute the adjoint for this operation}}
    %r = arith.remf %x, %x : f64
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
