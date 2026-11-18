// RUN: %eopt --enzyme-wrap="infn=f outfn= argTys=enzyme_dup retTys=enzyme_dupnoneed mode=ForwardMode" --verify-diagnostics %s

// A pass that does not check the keys up front reports a missing handler at
// the call.

module attributes {enzyme.forward_call_handlers = [[@g, "unregistered"]]} {
  func.func private @g(%x: f64) -> f64 {
    return %x : f64
  }
  func.func @f(%x: f64) -> f64 {
    // expected-error @below {{no forward call handler registered under 'unregistered'}}
    %y = func.call @g(%x) : (f64) -> f64
    return %y : f64
  }
}
