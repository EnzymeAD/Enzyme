// Forward-mode AD through a Fortran `function` whose result flows through a
// local `fir.alloca` result buffer. This is the HLFIR for
//
//   real function square(x); real,intent(in)::x; square = x*x; end
//
// The result `square` lives in a stack `fir.alloca` that is written via
// `hlfir.assign` and read back with `fir.load` for the `return`. For the
// tangent to reach the returned value, activity analysis must recognise the
// `!fir.ref` alloca as pointer-like memory (via AutoDiffTypeInterface::isMutable)
// so it is given a shadow buffer. d/dx (x*x) = 2*x, so the returned tangent is
// 2 * x * dx.
//
// REQUIRES: fir_enzyme_opt
// RUN: %fireopt --enzyme %s | FileCheck %s

func.func @_QPsquare(%arg0: !fir.ref<f32> {fir.bindc_name = "x", fir.read_only}) -> f32 {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca f32 {bindc_name = "square", uniq_name = "_QFsquareEsquare"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFsquareEsquare"}
    : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFsquareEx"}
    : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.load %3#0 : !fir.ref<f32>
  %5 = fir.load %3#0 : !fir.ref<f32>
  %6 = arith.mulf %4, %5 : f32
  hlfir.assign %6 to %2#0 : f32, !fir.ref<f32>
  %7 = fir.load %2#0 : !fir.ref<f32>
  return %7 : f32
}

func.func @dsquare(%x: !fir.ref<f32>, %dx: !fir.ref<f32>) -> f32 {
  %0 = enzyme.fwddiff @_QPsquare(%x, %dx) {
    activity = [#enzyme<activity enzyme_dup>],
    ret_activity = [#enzyme<activity enzyme_dupnoneed>]
  } : (!fir.ref<f32>, !fir.ref<f32>) -> (f32)
  return %0 : f32
}

// The dual forms the tangent 2*x*dx, stores it into the shadow result buffer,
// and returns the shadow load -- not a bare 0.0 constant.
// CHECK-LABEL: func.func private @fwddiffe_QPsquare
// CHECK: %[[shadowbuf:.+]] = fir.alloca f32
// CHECK: hlfir.declare %[[shadowbuf]]
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: %[[tan:.+]] = arith.addf
// CHECK: hlfir.assign %[[tan]] to %[[sdecl:.+]]#0 : f32, !fir.ref<f32>
// CHECK: %[[ret:.+]] = fir.load %[[sdecl]]#0 : !fir.ref<f32>
// CHECK: return %[[ret]] : f32
