module {

  func.func @mul(%a: f32, %b: f32) -> f32 {
    %0 = arith.mulf %a, %b : f32
    return %0 : f32
  }

  func.func @main() {
    %a = arith.constant 1.0 : f32
    %b = arith.constant 1.0 : f32

    %r, %tape = enzyme.autodiff_deferred_primal @mul(%a, %b) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (f32, f32) -> (f32, !enzyme.Tape)

    // ---

    %dres = arith.constant 1.0 : f32
    %da, %db = enzyme.autodiff_deferred_reverse @mul(%tape, %dres) {
      activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (!enzyme.Tape, f32) -> (f32, f32)

    return
  }

  func.func @mul_primal(%a: f32, %b: f32) -> (f32, !enzyme.Tape) {
    %cache0 = "enzyme.init"() : () -> !enzyme.Cache<f32>
    %cache1 = "enzyme.init"() : () -> !enzyme.Cache<f32>

    %0 = arith.mulf %a, %b : f32
    "enzyme.push"(%cache0, %a) : (!enzyme.Cache<f32>, f32) -> ()
    "enzyme.push"(%cache1, %a) : (!enzyme.Cache<f32>, f32) -> ()

    %tape = "enzyme_.new_tape"(%cache0, %cache1) : (!enzyme.Cache<f32>, !enzyme.Cache<f32>) -> !enzyme.Tape
    return %0, %tape : f32, !enzyme.Tape
  }

}
