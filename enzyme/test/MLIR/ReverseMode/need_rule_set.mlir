// RUN: %eopt %s --split-input-file --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --verify-diagnostics

module {
  enzyme.custom_reverse_rule @reverse_f {
    %cache_exp = "enzyme.init"() : () -> !enzyme.Cache<f32>
    %cache_arg1 = "enzyme.init"() : () -> !enzyme.Cache<f32>

    enzyme.custom_reverse_rule.augmented_primal (%arg0: f32, %arg1: f32) -> f32 {
      "enzyme.push"(%cache_arg1, %arg1) : (!enzyme.Cache<f32>, f32) -> ()
      %0 = math.exp %arg0 : f32
      "enzyme.push"(%cache_exp, %0) : (!enzyme.Cache<f32>, f32) -> ()
      %1 = arith.mulf %arg1, %0 : f32
      enzyme.yield %1 : f32
    }

    enzyme.custom_reverse_rule.reverse (%dres: f32) -> f32 {
      %exp = "enzyme.pop"(%cache_exp) : (!enzyme.Cache<f32>) -> f32
      %arg1 = "enzyme.pop"(%cache_arg1) : (!enzyme.Cache<f32>) -> f32
      %d1 = arith.mulf %dres, %arg1 : f32
      %darg0 = arith.mulf %exp, %d1 : f32
      enzyme.yield %darg0 : f32
    }

    enzyme.yield
  } attributes {
    activity=[#enzyme.activity<enzyme_active>,
              #enzyme.activity<enzyme_const>],
    ret_activity=[#enzyme.activity<enzyme_active>],
    function_type = (f32, f32) -> f32
  }

  func.func @f(%arg0:f32, %arg1: f32) -> f32 attributes {enzyme.custom_rule = @reverse_f} {
    %0 = math.exp %arg0 : f32
    %1 = arith.mulf %arg1, %0 : f32
    return %1 : f32
  }

  func.func @main(%arg0: f32, %arg1: f32) -> f32 {

    // expected-error @below {{could not find a rule with the right activity (rule activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_active>])}}
    %0 = func.call @f( %arg0, %arg1 ) : (f32, f32) -> f32

    return %0 : f32
  }
}

// -----

module {
  // expected-error @below {{a custom reverse rule needs one augmented primal and one reverse}}
  enzyme.custom_reverse_rule @missing_reverse {
    enzyme.custom_reverse_rule.augmented_primal (%x: f32) -> f32 {
      enzyme.yield %x : f32
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f32) -> f32
  }
  func.func @main(%x: f32, %y: f32) -> f32 {
    %r = arith.addf %x, %y : f32
    return %r : f32
  }
}

// -----

module {
  // expected-error @below {{a custom reverse rule needs one augmented primal and one reverse}}
  enzyme.custom_reverse_rule @missing_primal {
    enzyme.custom_reverse_rule.reverse (%dx: f32) -> f32 {
      enzyme.yield %dx : f32
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f32) -> f32
  }
  func.func @main(%x: f32, %y: f32) -> f32 {
    %r = arith.addf %x, %y : f32
    return %r : f32
  }
}

// -----

module {
  // expected-error @below {{a custom reverse rule needs one body block}}
  enzyme.custom_reverse_rule @empty_rule {
  } attributes {
    activity = [],
    ret_activity = [],
    function_type = () -> ()
  }
  func.func @main(%x: f32, %y: f32) -> f32 {
    %r = arith.addf %x, %y : f32
    return %r : f32
  }
}

// -----

module {
  enzyme.custom_reverse_rule @missing_pop {
    // expected-error @below {{a custom rule cache needs one push and one pop}}
    %cache = "enzyme.init"() : () -> !enzyme.Cache<f32>
    enzyme.custom_reverse_rule.augmented_primal (%x: f32) -> f32 {
      "enzyme.push"(%cache, %x) : (!enzyme.Cache<f32>, f32) -> ()
      enzyme.yield %x : f32
    }
    enzyme.custom_reverse_rule.reverse (%dx: f32) -> f32 {
      enzyme.yield %dx : f32
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f32) -> f32
  }
  func.func @main(%x: f32, %y: f32) -> f32 {
    %r = arith.addf %x, %y : f32
    return %r : f32
  }
}

// -----

module {
  // expected-error @below {{custom rule activities must match its function type}}
  enzyme.custom_reverse_rule @wrong_activity_count {
    enzyme.custom_reverse_rule.augmented_primal (%x: f32) -> f32 {
      enzyme.yield %x : f32
    }
    enzyme.custom_reverse_rule.reverse (%dx: f32) -> f32 {
      enzyme.yield %dx : f32
    }
    enzyme.yield
  } attributes {
    activity = [],
    ret_activity = [#enzyme.activity<enzyme_active>],
    function_type = (f32) -> f32
  }
  func.func @main(%x: f32, %y: f32) -> f32 {
    %r = arith.addf %x, %y : f32
    return %r : f32
  }
}

// -----

module {
  // expected-error @below {{unsupported custom rule return activity}}
  enzyme.custom_reverse_rule @unsupported_return {
    enzyme.custom_reverse_rule.augmented_primal (%x: f32) -> f32 {
      enzyme.yield %x : f32
    }
    enzyme.custom_reverse_rule.reverse (%dx: f32) -> f32 {
      enzyme.yield %dx : f32
    }
    enzyme.yield
  } attributes {
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_dup>],
    function_type = (f32) -> f32
  }
  func.func @main(%x: f32, %y: f32) -> f32 {
    %r = arith.addf %x, %y : f32
    return %r : f32
  }
}
