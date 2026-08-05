// Forward mode through FIR's by-reference memory model. This is the HLFIR for
//
//   subroutine sq(x, r);  real,intent(in)::x; real,intent(out)::r; r = x*x; end
//
// whose !fir.ref<f32> arguments are read and written by fir.load / hlfir.assign.
// d/dx (x*x) = 2*x, so the shadow output dr receives 2 * x * dx.
//
// REQUIRES: fir_enzyme_plugin
// RUN: %fir_enzyme --pass-pipeline='builtin.module(enzyme)' %s | FileCheck %s

func.func @_QPsq(%arg0: !fir.ref<f32> {fir.bindc_name = "x", fir.read_only},
                 %arg1: !fir.ref<f32> {fir.bindc_name = "r"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "r"}
    : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "x"}
    : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.load %2#0 : !fir.ref<f32>
  %4 = fir.load %2#0 : !fir.ref<f32>
  %5 = arith.mulf %3, %4 : f32
  hlfir.assign %5 to %1#0 : f32, !fir.ref<f32>
  return
}

func.func @dsq(%x: !fir.ref<f32>, %dx: !fir.ref<f32>,
               %r: !fir.ref<f32>, %dr: !fir.ref<f32>) {
  enzyme.fwddiff @_QPsq(%x, %dx, %r, %dr) {
    activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (!fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>) -> ()
  return
}

// The dual loads the shadow dx and primal x, forms the tangent 2*x*dx, and
// stores it through the shadow output r.
// CHECK-LABEL: func.func private @fwddiffe_QPsq
// CHECK: %[[dx0:.+]] = fir.load %{{.*}} : !fir.ref<f32>
// CHECK: %[[x0:.+]] = fir.load %{{.*}} : !fir.ref<f32>
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: %[[tan:.+]] = arith.addf
// CHECK: hlfir.assign %[[tan]] to %{{.*}} : f32, !fir.ref<f32>
