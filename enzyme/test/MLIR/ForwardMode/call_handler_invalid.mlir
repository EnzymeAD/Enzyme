// RUN: %eopt --test-register-forward-call-handlers="handlers=norule" --enzyme --split-input-file --verify-diagnostics %s

// A handler's failure is the derivative's failure.

module attributes {enzyme.forward_call_handlers = [[@norule, "norule"]]} {
  func.func private @norule(%x: f64) -> f64 {
    return %x : f64
  }
  func.func @f(%x: f64) -> f64 {
    // expected-error @below {{test forward call handler: no function @norule_fwd_dup}}
    %y = func.call @norule(%x) : (f64) -> f64
    return %y : f64
  }
  func.func @df(%x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @f(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (f64, f64) -> f64
    return %r : f64
  }
}

// -----

// Every listed key needs a registered handler, also for a function that is
// not called.

// expected-error @below {{no forward call handler registered under 'unregistered'}}
module attributes {enzyme.forward_call_handlers = [[@g, "unregistered"]]} {
  func.func private @g(%x: f64) -> f64 {
    return %x : f64
  }
}

// -----

// expected-error @below {{'enzyme.forward_call_handlers' must be an array attribute}}
module attributes {enzyme.forward_call_handlers = "norule"} {
}

// -----

// expected-error @below {{'enzyme.forward_call_handlers' entry must be a [@function, "key"] pair, got [@f]}}
module attributes {enzyme.forward_call_handlers = [[@f]]} {
}
