! `flang -fc1 -load` of FlangEnzymeMLIR loads it into flang, and code comes
! through flang's pipeline unchanged.
!
! REQUIRES: flang_enzyme_mlir
! RUN: %flang_enzyme -emit-fir %s -o - | FileCheck %s

module plain
  implicit none
  public
contains
  subroutine scale(x, r)
    real, intent(in)  :: x
    real, intent(out) :: r
    r = x * 2.0
  end subroutine scale
end module plain

! CHECK-LABEL: func.func @_QMplainPscale
! CHECK: arith.mulf
