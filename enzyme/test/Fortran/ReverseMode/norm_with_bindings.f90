! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O0 -S -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O1 -S -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O2 -S -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O3 -S -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: %if flangenzyme %{ %fc -O0 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}
! RUN: %if flangenzyme %{ %fc -O2 %loadFortran %loadFlangEnzyme %s -o %t2 && %t2 | FileCheck %s %}

! NOTE: This test is only configured to run with the flang compiler
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the indirection involved in the enzyme_autodiff binding

program main
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  implicit none
  integer, parameter :: n = 1000000
  integer, parameter :: initial_value = 20
  real :: x(n), dx(n)
  real :: y(n), dy(n), yp

  x(:) = initial_value
  dy(:) = 1.0

  call norm(n, x, y)

  ! Rescale the output to avoid compiler-specific output formatting
  yp = y(n) * 1.0e+06
  write(*,"(f6.4)") yp

  dx(:) = 0.0
  call enzyme_autodiff(norm, enzyme_const, n, &
                       enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*,"(f6.4)") dy(n)

contains

  subroutine norm(n, x, y)
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: x
    real, dimension(n), intent(out) :: y
    real :: s
    integer :: i
    ! TODO: Use the `sum` intrinsic. Requires accounting for `_FortranASumReal4`
    !       for this to work at -O0
    ! TODO: Use an array assignment. Requires accounting for `_FortranAAssign`
    !       for this to work at -O0
    ! y(:) = x / sum(x)
    s = 0.0
    do i = 1, n
      s = s + x(i)
    end do
    do i = 1, n
      y(i) = x(i) / s
    end do
  end subroutine norm

end program main

! CHECK: 1.0000
! CHECK-NEXT: 0.0000
