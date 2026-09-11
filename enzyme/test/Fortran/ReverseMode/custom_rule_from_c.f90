! REQUIRES: fortran
! REQUIRES: clang
! UNSUPPORTED: ifx

! A custom derivative rule for a Fortran routine, registered from a C
! translation unit and linked in at the IR level.
!
! Each translation unit is compiled to bitcode separately, llvm-link merges
! them, and only then is Enzyme run -- the registration global has to be in the
! same module as the primal for preserve-nvvm to attach the rule to it.  This
! is why the %loadFlangEnzyme route is not exercised here: -fpass-plugin runs
! Enzyme inside flang, per translation unit, where the C registration is not
! yet visible.

! RUN: %clang -flto -O0 -c %S/Inputs/custom_square_rule.c -o %t.rule.bc
! RUN: %fc -flto -O0 -c %S/Inputs/custom_square_primal.f90 -o %t.primal.bc
! RUN: %fc -flto -O0 -c %loadFortran %s -o %t.main.bc
! RUN: llvm-link %t.main.bc %t.primal.bc %t.rule.bc -o %t.linked.bc
! RUN: %opt %newLoadEnzyme -passes="preserve-nvvm,enzyme" %t.linked.bc -o %t.ll
! RUN: %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s

! RUN: %clang -flto -O2 -c %S/Inputs/custom_square_rule.c -o %t2.rule.bc
! RUN: %fc -flto -O2 -c %S/Inputs/custom_square_primal.f90 -o %t2.primal.bc
! RUN: %fc -flto -O2 -c %loadFortran %s -o %t2.main.bc
! RUN: llvm-link %t2.main.bc %t2.primal.bc %t2.rule.bc -o %t2.linked.bc
! RUN: %opt %newLoadEnzyme -passes="preserve-nvvm,enzyme" %t2.linked.bc -o %t2.ll
! RUN: %fc -flto -O2 %t2.ll -o %t3 && %t3 | FileCheck %s

module custom_rule_driver
  use iso_c_binding, only: c_double, c_int
  implicit none

  interface
    ! The routine whose derivative is overridden; defined in
    ! Inputs/custom_square_primal.f90.
    subroutine fsquare(x, y) bind(C, name="fsquare")
      import :: c_double
      real(c_double), intent(in) :: x
      real(c_double), intent(out) :: y
    end subroutine fsquare

    ! Call counters owned by Inputs/custom_square_rule.c.
    function augment_calls() bind(C, name="enzyme_augment_calls")
      import :: c_int
      integer(c_int) :: augment_calls
    end function augment_calls

    function gradient_calls() bind(C, name="enzyme_gradient_calls")
      import :: c_int
      integer(c_int) :: gradient_calls
    end function gradient_calls
  end interface

contains

  ! Enzyme differentiates this.  The call to fsquare is where the registered
  ! rule takes over.
  function wrapper(x) result(y)
    real(c_double), intent(in) :: x
    real(c_double) :: y
    call fsquare(x, y)
  end function wrapper

end module custom_rule_driver

program main
  use custom_rule_driver, only: wrapper, augment_calls, gradient_calls
  use enzyme, only: enzyme_autodiff
  use iso_c_binding, only: c_double
  implicit none
  real(c_double) :: x, dx

  x = 3.0_c_double
  dx = 0.0_c_double
  call enzyme_autodiff(wrapper, x, dx)

  ! The true derivative of x**2 at x = 3 is 6.  Seeing 13 instead is what
  ! proves the rule registered from the C translation unit was the one used.
  write(*,"(a,f0.3)") "dx = ", dx
  write(*,"(a,i0)") "augment calls = ", augment_calls()
  write(*,"(a,i0)") "gradient calls = ", gradient_calls()
end program main

! CHECK: dx = 13.000
! CHECK-NEXT: augment calls = 1
! CHECK-NEXT: gradient calls = 1
