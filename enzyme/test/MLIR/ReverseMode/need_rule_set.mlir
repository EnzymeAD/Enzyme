// RUN: %eopt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --verify-diagnostics

module {
  enzyme.custom_reverse_rule @reverse_f {
    %cache_exp = "enzyme.init"() : () -> !enzyme.Cache<f32>
    %cache_arg1 = "enzyme.init"() : () -> !enzyme.Cache<f32>

    enzyme.custom_reverse_rule.augmented_primal (%arg0: f32, %arg1: f32) -> f32 {
      "enzyme.push"(%cache_arg1, %arg1) : (!enzyme.Cache<f32>, f32) -> ()
      %0 = math.exp %arg0 : f32
      "enzyme.push"(%cache_exp, %0) : (!enzyme.Cache<f32>, f32) -> ()
      %1 = arith.mulf %arg1, %arg0 : f32
      enzyme.yield %1 : f32
    }

    enzyme.custom_reverse_rule.augmented_primal (%dres: f32) -> f32 {
      %exp = "enzyme.pop"(%cache_exp) : (!enzyme.Cache<f32>) -> f32
      %arg1 = "enzyme.pop"(%cache_exp) : (!enzyme.Cache<f32>) -> f32
      %d1 = arith.mulf %dres, %arg1 : f32
      %darg0 = arith.mulf %exp, %d1 : f32
      enzyme.yield %darg0 : f32
    }

    enzyme.yield
  } attributes {
    activity=[#enzyme<activity enzyme_active>,
              #enzyme<activity enzyme_const>],
    ret_activity=[#enzyme<activity enzyme_active>],
    function_type = (f32, f32) -> f32
  }

  func.func @f(%arg0:f32, %arg1: f32) -> f32 attributes {enzyme.custom_rule = @reverse_f} {
    %0 = math.exp %arg0 : f32
    %1 = arith.mulf %arg1, %arg0 : f32
    return %1 : f32
  }

  func.func @main(%arg0: f32, %arg1: f32) -> f32 {

    // expected-error @below {{could not find a rule with the right activity (rule activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>], ret_activity=[#enzyme<activity enzyme_active>])}}
    %0 = func.call @f( %arg0, %arg1 ) : (f32, f32) -> f32

    return %0 : f32
  }
}
