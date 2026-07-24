! End-to-end MLIR path with Fortran as the input: flang lowers the MATMUL
! intrinsic to a first-class hlfir.matmul op, and fir-enzyme-opt (fir-opt with
! Enzyme linked in) ingests that HLFIR and runs the `enzyme` differentiation
! pass over it in an FIR/HLFIR-aware context.
!
! This exercises the pipeline wiring end-to-end. Differentiating a whole Fortran
! function additionally needs !fir.ref active-memory support and rules for the
! surrounding memory ops (hlfir.declare/assign/destroy, fir.load); the Tier-1
! hlfir.matmul rules themselves are covered by matmul_fwd.mlir / matmul_rev.mlir.
!
! REQUIRES: flang_new, fir_enzyme_opt
! RUN: %flang_fc1 -emit-hlfir %s -o - | %fireopt --enzyme | FileCheck %s

function mm(a, b) result(c)
  real, intent(in) :: a(2,2), b(2,2)
  real :: c(2,2)
  c = matmul(a, b)
end function mm

! The MATMUL intrinsic lowered to a first-class hlfir.matmul, carried through
! the Enzyme pass unchanged (no enzyme.autodiff driver in this module).
! CHECK-LABEL: func.func @_QPmm
! CHECK: hlfir.matmul %{{.*}} %{{.*}} : (!fir.ref<!fir.array<2x2xf32>>, !fir.ref<!fir.array<2x2xf32>>) -> !hlfir.expr<2x2xf32>
