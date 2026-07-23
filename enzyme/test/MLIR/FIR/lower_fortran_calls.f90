! The enzyme-lower-fortran-calls pass rewrites a Fortran Enzyme differentiation
! hook call into a first-class enzyme.fwddiff op at the HLFIR stage: the callee
! is recovered from the fir.emboxproc/fir.address_of, and the enzyme_dup /
! enzyme_const activity markers select per-argument activity (enzyme_dup pairs
! the primal with its shadow; enzyme_const takes the primal alone).
!
! REQUIRES: flang_new, fir_enzyme_opt
! RUN: %flang_fc1 -emit-hlfir %s -o - | %fireopt --enzyme-lower-fortran-calls | FileCheck %s

module marks
  integer, bind(C, name="enzyme_dup")   :: enzyme_dup
  integer, bind(C, name="enzyme_const") :: enzyme_const
end module

real function square(x, y)
  real, intent(in) :: x, y
  square = x * x + y
end function

subroutine driver(x, dx, y, r)
  use marks
  real, intent(in)  :: x, dx, y
  real, intent(out) :: r
  real, external    :: square
  real, external    :: f__enzyme_fwddiff
  r = f__enzyme_fwddiff(square, enzyme_dup, x, dx, enzyme_const, y)
end subroutine

! CHECK-LABEL: func.func @_QPdriver
! x active (dup, with shadow dx), y inactive (const); markers are dropped.
! CHECK: enzyme.fwddiff @_QPsquare(
! CHECK-SAME: activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>]
! CHECK-SAME: ret_activity = [#enzyme<activity enzyme_dupnoneed>]
! CHECK-NOT: fir.call @_QPf__enzyme_fwddiff
