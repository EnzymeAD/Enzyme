// Reverse-mode AD through a Fortran `function` whose result flows through a
// local `fir.alloca` result buffer. This is the HLFIR for
//
//   real function square(x); real,intent(in)::x; square = x*x; end
//
// The result `square` lives in a stack `fir.alloca` written via `hlfir.assign`
// and read back with `fir.load` for the `return`. In reverse mode the return
// adjoint seeds the shadow result buffer, propagates back through the assign
// and the two loads of x, and accumulates 2*x*seed into the shadow x.
//
// REQUIRES: fir_enzyme_opt
// RUN: %fireopt %s --enzyme-wrap="infn=square outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

func.func @square(%x: !fir.ref<f32>) -> f32 {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca f32 {bindc_name = "square", uniq_name = "_QFsquareEsquare"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFsquareEsquare"}
    : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3:2 = hlfir.declare %x dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFsquareEx"}
    : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.load %3#0 : !fir.ref<f32>
  %5 = fir.load %3#0 : !fir.ref<f32>
  %6 = arith.mulf %4, %5 : f32
  hlfir.assign %6 to %2#0 : f32, !fir.ref<f32>
  %7 = fir.load %2#0 : !fir.ref<f32>
  return %7 : f32
}

// The differentiated function takes the shadow x and the return adjoint, and
// drives the tangent 2*x*seed back into the shadow x. A shadow (zeroed) result
// buffer mirrors the primal alloca.
// CHECK-LABEL: func.func @square(
// CHECK-SAME:    %[[X:.*]]: !fir.ref<f32>, %[[DX:.*]]: !fir.ref<f32>, %[[SEED:.*]]: f32) {
// Shadow result buffer (zeroed) + its declare.
// CHECK: %[[SBUF:.*]] = fir.alloca f32
// CHECK: fir.store %{{.*}} to %[[SBUF]] : !fir.ref<f32>
// CHECK: %[[SBUFD:.*]]:2 = hlfir.declare %[[SBUF]]
// CHECK: %[[DXD:.*]]:2 = hlfir.declare %[[DX]]
// The return adjoint is accumulated into the shadow result buffer (fir.load rev).
// CHECK: fir.store %{{.*}} to %[[SBUFD]]#0 : !fir.ref<f32>
// The assign adjoint pops the seed and zeroes the shadow result buffer.
// CHECK: fir.store %{{.*}} to %[[SBUFD]]#0 : !fir.ref<f32>
// Final gradient accumulates into the shadow x.
// CHECK: fir.store %{{.*}} to %[[DXD]]#0 : !fir.ref<f32>
// CHECK: fir.store %{{.*}} to %[[DXD]]#0 : !fir.ref<f32>
