// RUN: %eopt --test-register-forward-call-handlers="handlers=norule" --enzyme --split-input-file --verify-diagnostics %s

// A handler's failure is the derivative's failure.

module {
  func.func private @norule(%x: f64) -> f64 attributes {enzyme_math = "norule"} {
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

// An indirect call names no callee to look a handler up for.

module {
  llvm.func @ind(%fp: !llvm.ptr, %x: f64) -> f64 {
    // expected-error @below {{could not find the callee of}}
    %y = llvm.call %fp(%x) : !llvm.ptr, (f64) -> f64
    llvm.return %y : f64
  }
  llvm.func @dind(%fp: !llvm.ptr, %x: f64, %dx: f64) -> f64 {
    %r = enzyme.fwddiff @ind(%fp, %x, %dx) <{ activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!llvm.ptr, f64, f64) -> f64
    llvm.return %r : f64
  }
}
