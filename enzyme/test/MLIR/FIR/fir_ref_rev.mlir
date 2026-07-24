// Reverse-mode AD through FIR's by-reference memory model (!fir.ref). This is
// the HLFIR for the Fortran subroutine
//
//   subroutine sq(x, r);  real,intent(in)::x; real,intent(out)::r; r = x*x; end
//
// whose arguments are !fir.ref<f32> loaded/stored via fir.load / hlfir.assign.
// In reverse mode the adjoint of the output r flows back through the shadow
// buffers: given r̄ (dr), the input adjoint is dx += 2*x*dr.
//
// REQUIRES: fir_enzyme_opt
// RUN: %fireopt %s --enzyme-wrap="infn=sq outfn= argTys=enzyme_dup,enzyme_dup retTys= mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

func.func @sq(%x: !fir.ref<f32>, %r: !fir.ref<f32>) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %r dummy_scope %0 {uniq_name = "r"}
    : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2:2 = hlfir.declare %x dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "x"}
    : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.load %2#0 : !fir.ref<f32>
  %4 = fir.load %2#0 : !fir.ref<f32>
  %5 = arith.mulf %3, %4 : f32
  hlfir.assign %5 to %1#0 : f32, !fir.ref<f32>
  return
}

// The reverse sweep pops the output adjoint from the shadow r (dr), zeroes it
// (the hlfir.assign adjoint), backpropagates through the two loads of x, and
// accumulates 2*x*dr into the shadow x (dx) via load/add/store. Shadow refs are
// captured by SSA use rather than by argument order.
// CHECK-LABEL: func.func @sq(
// Primal x loads.
// CHECK: %[[X0:.*]] = fir.load %{{.*}}#0 : !fir.ref<f32>
// CHECK: %[[X1:.*]] = fir.load %{{.*}}#0 : !fir.ref<f32>
// Pop + zero the output adjoint (hlfir.assign reverse) from the shadow r (dr).
// CHECK: %[[G:.*]] = fir.load %[[DRD:.*]]#0 : !fir.ref<f32>
// CHECK: %[[SEED:.*]] = arith.addf %[[G]], %{{.*}} : f32
// CHECK: fir.store %{{.*}} to %[[DRD]]#0 : !fir.ref<f32>
// Backprop through mul: contributions to each loaded x.
// CHECK: arith.mulf %[[SEED]], %[[X1]] : f32
// CHECK: arith.mulf %[[SEED]], %[[X0]] : f32
// Accumulate both into the shadow x (dx) via load/add/store.
// CHECK: fir.load %[[DXD:.*]]#0 : !fir.ref<f32>
// CHECK: arith.addf
// CHECK: fir.store %{{.*}} to %[[DXD]]#0 : !fir.ref<f32>
// CHECK: fir.load %[[DXD]]#0 : !fir.ref<f32>
// CHECK: arith.addf
// CHECK: fir.store %{{.*}} to %[[DXD]]#0 : !fir.ref<f32>
