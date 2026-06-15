! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O0 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O0 -S -o %t.ll && %fc -flto -O0 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O1 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O1 -S -o %t.ll && %fc -flto -O1 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O2 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O2 -S -o %t.ll && %fc -flto -O2 %t.ll -o %t1 && %t1 | FileCheck %s; fi
! RUN: if [[ %llvmver -ge 13 && %fc != ifx ]]; then %fc -flto -O3 -c %loadFortran %s -o /dev/stdout | %opt %loadEnzyme %enzyme -o /dev/stdout | %opt -O3 -S -o %t.ll && %fc -flto -O3 %t.ll -o %t1 && %t1 | FileCheck %s; fi

! NOTE: This test is only configured to run with the flang compiler
!       For it to work with the ifx compiler we will need to figure out how to
!       handle the indirection involved in the enzyme_autodiff binding

module math
contains
  subroutine norm(n, x, y)
    integer, intent(in) :: n
    real, dimension(n), intent(in) :: x
    real, dimension(n), intent(out) :: y
    y(:) = x / sum(x)
  end subroutine norm
end module math

program app
  use enzyme, only: enzyme_const, enzyme_dup, enzyme_autodiff
  use math, only: norm
  implicit none
  integer, parameter :: n = 1000000
  integer, parameter :: initial_value = 20
  real :: x(n), dx(n)
  real :: y(n), dy(n)

  x(:) = initial_value
  dy(:) = 1.0

  call norm(n, x, y)
  write(*,"(es0.0)") y(n)

  dx(:) = 0.0
  call enzyme_autodiff(norm, enzyme_const, n, &
                       enzyme_dup, x, dx, enzyme_dup, y, dy)
  write(*,"(es0.0)") dy(n)
end program app

! CHECK: 1.E-06
! CHECK-NEXT: 0.
