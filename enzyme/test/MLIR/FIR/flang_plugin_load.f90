! Loading FlangEnzymeMLIR into `flang -fc1` registers the Enzyme passes into
! flang's own HLFIR-to-FIR pipeline. Code with no differentiation hook must come
! through that pipeline unchanged, which is what exercises the registration
! itself: the plugin loads, the extension point fires, and both added passes run
! as no-ops.
!
! REQUIRES: flang_enzyme_mlir
! RUN: %flang_enzyme -emit-fir %s -o - | FileCheck %s

subroutine scale(x, r)
  real, intent(in)  :: x
  real, intent(out) :: r
  r = x * 2.0
end subroutine

! CHECK-LABEL: func.func @_QPscale
! CHECK: arith.mulf
! CHECK-NOT: enzyme.
