module {

  func.func @mul(%a: f32, %b: f32) -> f32 {
    %0 = arith.mulf %a, %b : f32
    return %0 : f32
  }

  // Split mode
  func.func @main() {

    %a = arith.constant 1.0 : f32
    %b = arith.constant 1.0 : f32

    %r, %tape = enzyme.autodiff_split_mode.primal @mul(%a, %b) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (f32, f32) -> (f32, !enzyme.Tape)

    // ---

    %dres = arith.constant 1.0 : f32
    %da, %db = enzyme.autodiff_split_mode.reverse @mul(%dres, %tape) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (f32, !enzyme.Tape) -> (f32, f32)

    return
  }

  enzyme.custom_reverse_rule @reverse_f {
    %0 = "enzyme.init"() : () -> !enzyme.Cache<f32>
    %1 = "enzyme.init"() : () -> !enzyme.Cache<f32>

    enzyme.custom_reverse_rule.augmented_primal (%a: f32, %b: f32) -> f32 {
      "enzyme.push"(%0, %a) : (!enzyme.Cache<f32>, f32) -> ()
      "enzyme.push"(%1, %b) : (!enzyme.Cache<f32>, f32) -> ()

      %res = arith.mulf %a, %b : f32

      enzyme.yield %res : f32
    }

    enzyme.custom_reverse_rule.reverse (%dres: f32) -> (f32, f32) {
      %a = "enzyme.pop"(%0) : (!enzyme.Cache<f32>) -> f32
      %b = "enzyme.pop"(%1) : (!enzyme.Cache<f32>) -> f32

      %da = arith.mulf %b, %dres : f32
      %db = arith.mulf %a, %dres : f32

      enzyme.yield %da, %db : f32, f32
    }

    enzyme.yield
  } attributes {
      activity=[#enzyme<activity enzyme_active>,
                #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>],
      function_type = (f32, f32) -> (f32)
  }

  func.func @custom_rule_call(%a: f32, %b: f32, %dres_in: f32) -> (f32, f32) {
    %res, %tape = enzyme.call_augmented_primal @reverse_f(%a, %b) : (f32, f32) -> (f32, !enzyme.Tape)

    %res_g = "enzyme.init"() : () -> (!enzyme.Gradient<f32>)
    %zero = arith.constant 0.0 : f32

    "enzyme.set"(%res_g, %dres_in) : (!enzyme.Gradient<f32>, f32) -> ()
    %dres = "enzyme.get"(%res_g) : (!enzyme.Gradient<f32>) -> f32
    %da, %db = enzyme.call_custom_reverse @reverse_f(%dres, %tape) : (f32, !enzyme.Tape) -> (f32, f32)
    return %da, %db : f32, f32
  }

  enzyme.custom_reverse_rule @exp_f32 {
    %cache = "enzyme.init"() : () -> !enzyme.Cache<f32>
    enzyme.custom_reverse_rule.augmented_primal (%arg0: f32) {
      %res = math.exp %arg0 : f32
      "enzyme.push"(%cache, %res) : (!enzyme.Cache<f32>, f32) -> ()
      enzyme.yield %res : f32
    }
    enzyme.custom_reverse_rule.reverse (%dres: f32) -> f32 {
      %res = "enzyme.pop"(%cache) : (!enzyme.Cache<f32>) -> (f32)
      %darg0 = arith.mulf %dres, %res : f32
      enzyme.yield %darg0 : f32
    }
    enzyme.yield
  } attributes {
      activity=[#enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>],
      function_type = (f32) -> (f32)
  }

  func.func @f_dup(%a: !llvm.ptr) -> f32 {
    %0 = llvm.load %a : !llvm.ptr -> f32
    return %0 : f32
  }

  func.func @ff_dup(%a: !llvm.ptr, %b: !llvm.ptr) -> f32 {
    %tape_cache = "enzyme.init"() : () -> !enzyme.Cache<!enzyme.Tape>

    %r, %tape = enzyme.autodiff_split_mode.primal @f_dup(%a, %b) {
      activity=[#enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (!llvm.ptr, !llvm.ptr) -> (f32, !enzyme.Tape)
    "enzyme.push"(%tape_cache, %tape) : (!enzyme.Cache<!enzyme.Tape>, !enzyme.Tape) -> ()

    //

    %popped_tape = "enzyme.pop"(%tape_cache) : (!enzyme.Cache<!enzyme.Tape>) -> !enzyme.Tape
    %dres = arith.constant 1.0 : f32
    enzyme.autodiff_split_mode.reverse @f_dup(%dres, %popped_tape) {
      activity=[#enzyme<activity enzyme_dup>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (f32, !enzyme.Tape) -> ()

    return %r : f32
  }

  // TODO: split mode is implemented using custom_reverse_rule + AD

  // func.func @mul_primal(%a: f32, %b: f32) -> (f32, !enzyme.Tape) {
  //   %cache0 = "enzyme.init"() : () -> !enzyme.Cache<f32>
  //   %cache1 = "enzyme.init"() : () -> !enzyme.Cache<f32>

  //   %0 = arith.mulf %a, %b : f32
  //   "enzyme.push"(%cache0, %a) : (!enzyme.Cache<f32>, f32) -> ()
  //   "enzyme.push"(%cache1, %a) : (!enzyme.Cache<f32>, f32) -> ()

  //   %tape = "enzyme_.new_tape"(%cache0, %cache1) : (!enzyme.Cache<f32>, !enzyme.Cache<f32>) -> !enzyme.Tape
  //   return %0, %tape : f32, !enzyme.Tape
  // }

}
